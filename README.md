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
git tag v0.2.0
git push origin v0.2.0
```

Before the first release, configure this repository as a Trusted Publisher in
the `PyPI` project settings and set the workflow environment to `pypi` if `PyPI`
prompts for it. For crates.io, create an API token and store it in GitHub as
`CARGO_REGISTRY_TOKEN`.

Minimal Python usage:

```python
from pathlib import Path

import polars as pl
import tokmat_polars

plugin_path = Path(tokmat_polars.__file__).parent

expr = pl.plugins.register_plugin_function(
    plugin_path=plugin_path,
    function_name="tokenize_expr",
    args=pl.col("address"),
    kwargs={"model_path": "/path/to/model"},
    use_abs_path=True,
)
```

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
