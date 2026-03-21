# tokmat-polars

Standalone Polars integration crate for `tokmat`.

This crate depends on the published `tokmat` package from crates.io and is
intended to provide the dataframe-facing plugin layer around the core parser.

## Python packaging

`tokmat-polars` can also be built and published as a Python package via
`maturin`. The Rust crate exposes a `PyO3` extension module named
`tokmat_polars`, and Polars can load the compiled plugin functions from that
module path.

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
