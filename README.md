# tokmat-polars

Standalone Polars integration crate for `tokmat`.

This crate depends on the published `tokmat` package from crates.io and is
intended to provide the dataframe-facing plugin layer around the core parser.

## Rust usage

`tokmat-polars` can also be used directly from Rust as a normal library crate.
That path is useful when you want to build `Series` values in Rust and reuse the
same tokenization and extraction logic without going through Python.

```rust
use polars::prelude::*;
use tokmat_polars::TokmatPolars;

let plugin = TokmatPolars::from_model_path("tests/fixtures/model_1")?;
let input = Series::new("address".into(), ["123 MAIN ST"]);
let tokenized = plugin.tokenize_series(&input)?;
let extracted = plugin.extract_series(
    &tokenized,
    "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>",
)?;
# let _ = extracted;
# Ok::<(), PolarsError>(())
```

## Python packaging

`tokmat-polars` can also be built and published as a Python package via
`maturin`. The Rust crate exposes a `PyO3` extension module named
`tokmat_polars`, and Polars can load the compiled plugin functions from that
module path. Python support starts at 3.12.

Typical local workflow:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip maturin pytest polars
maturin develop
pytest -q
```

## Release workflow

`PyPI` releases are published from `GitHub Actions` using `Trusted Publishing`. The
release workflow builds wheels and an sdist on tag pushes that match `v*`, then
uploads them through `pypa/gh-action-pypi-publish`. The same tag also publishes
the Rust crate to crates.io using a `CARGO_REGISTRY_TOKEN` GitHub secret.

Release steps:

```bash
git tag v0.3.0
git push origin v0.3.0
```

Before the first release, configure this repository as a Trusted Publisher in
the `PyPI` project settings and set the workflow environment to `pypi` if `PyPI`
prompts for it. For crates.io, create an API token and store it in GitHub as
`CARGO_REGISTRY_TOKEN`.

## Python usage

Use the wrapper functions — `tokenize`, `extract`, `encode_class_ids`. They
register the plugin expressions with `is_elementwise=True`, so the Polars engine
can stream them and parallelize across chunks on its own thread pool:

```python
import polars as pl
import tokmat_polars as tk

MODEL = "/path/to/model"
PATTERN = "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>"

lf = pl.LazyFrame({"address": ["ATTN 123 MAIN ST"]})

# tokenize -> struct(raw_value, tokens, types, classes); unnest for flat columns
tok = lf.select(tk.tokenize(pl.col("address"), model_path=MODEL).alias("t")).unnest("t")

# extract -> struct(capture fields..., complement)
parsed = lf.select(
    tk.extract(pl.col("address"), model_path=MODEL, pattern=PATTERN, mode="any").alias("p")
).unnest("p")
```

### Use the streaming engine for large data

Because the expressions are elementwise, the streaming engine parallelizes them
and keeps memory bounded. On a 1M-row benchmark (`benchmarks/bench_polars_engine.py`),
extraction is **~7× faster** under streaming and uses near-zero extra memory:

```python
parsed.collect(engine="streaming")   # parallel + bounded memory
```

| 1M rows | in-memory | streaming |
| --- | --- | --- |
| `tokenize` | 4.7 s, +298 MB | 4.3 s, **+43 MB** |
| `extract`  | 35.6 s | **4.9 s**, +0 MB |

`is_elementwise=False` (raw `register_plugin_function` without the flag) forfeits
this — streaming can't engage and extraction stays at ~36 s. The wrapper sets the
flag for you; prefer it over registering the plugin by hand.

The word definition is also exposed: `tk.model_word_definition(path)`,
`tk.get_word_definition()`, `tk.set_word_definition(chars)`.

## Plugin API

`tokmat-polars` exposes two Polars plugin functions:

- `tokenize_expr`
- `extract_expr`

### `tokenize_expr`

Required kwargs:

- `model_path`

Returns a struct column with:

- `raw_value`
- `tokens`
- `types`
- `classes`

### `extract_expr`

Required kwargs:

- `model_path`
- `pattern`

Optional kwargs:

- `mode`

Supported `mode` values:

- `whole` (default)
- `start`
- `end`
- `any`

When `extract_expr` receives a tokenized struct column, it uses the embedded
`raw_value` field when computing complements. This preserves `any`-mode
behavior and keeps complement output aligned with the original text rather than
with a placeholder reconstruction.

Example:

```python
parsed = pl.DataFrame({"address": ["ATTN 123 MAIN ST"]}).select(
    pl.plugins.register_plugin_function(
        plugin_path=plugin_path,
        function_name="extract_expr",
        args=pl.col("address"),
        kwargs={
            "model_path": "/path/to/model",
            "pattern": "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>",
            "mode": "any",
        },
        use_abs_path=True,
    ).alias("parsed")
).unnest("parsed")
```

This returns capture fields plus a `complement` column. In the example above,
the complement contains `"ATTN "` because the TEL pattern matches only the
embedded address portion.
