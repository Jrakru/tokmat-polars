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
