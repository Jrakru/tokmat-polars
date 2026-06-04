"""tokmat-polars: Polars expression plugin for tokmat address parsing.

The user-facing API is the :func:`tokenize`, :func:`extract`, and
:func:`encode_class_ids` helpers. They register the compiled plugin expressions
with ``is_elementwise=True`` so the Polars engine can stream them and parallelize
them across chunks on its own thread pool (rather than the plugin re-parallelizing
internally).

Example
-------
>>> import polars as pl
>>> import tokmat_polars as tk
>>> lf = pl.LazyFrame({"address": ["123 MAIN ST"]})
>>> out = lf.select(
...     tk.tokenize(pl.col("address"), model_path="/path/to/model").alias("tok")
... ).unnest("tok")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from tokmat_polars import _tokmat_polars
from tokmat_polars._tokmat_polars import (
    get_word_definition,
    model_word_definition,
    set_word_definition,
)

if TYPE_CHECKING:
    import polars as pl
    from polars._typing import IntoExpr

__all__ = [
    "tokenize",
    "extract",
    "encode_class_ids",
    "model_word_definition",
    "get_word_definition",
    "set_word_definition",
]

# Absolute path to the compiled extension that carries the plugin expression
# symbols (`tokenize_expr`, `extract_expr`, `encode_class_ids_expr`).
_PLUGIN_PATH = Path(_tokmat_polars.__file__)

ListOutput = Literal["string", "categorical", "enum"]
MatchMode = Literal["whole", "start", "end", "any"]


def tokenize(
    expr: IntoExpr,
    *,
    model_path: str | Path,
    include_raw_value: bool = True,
    include_types: bool = True,
    include_classes: bool = True,
    include_type_ids: bool = False,
    include_class_ids: bool = False,
    token_output: ListOutput = "string",
    type_output: ListOutput = "string",
    class_output: ListOutput = "string",
) -> pl.Expr:
    """Tokenize a String column into a struct (``raw_value``/``tokens``/``types``/...).

    Elementwise: each row is tokenized independently, so the engine may stream and
    parallelize it. Use ``.unnest(...)`` on the result to get flat columns.
    """
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="tokenize_expr",
        args=expr,
        kwargs={
            "model_path": str(model_path),
            "include_raw_value": include_raw_value,
            "include_types": include_types,
            "include_classes": include_classes,
            "include_type_ids": include_type_ids,
            "include_class_ids": include_class_ids,
            "token_output": token_output,
            "type_output": type_output,
            "class_output": class_output,
        },
        is_elementwise=True,
        use_abs_path=True,
    )


def extract(
    expr: IntoExpr,
    *,
    model_path: str | Path,
    pattern: str,
    mode: MatchMode = "whole",
) -> pl.Expr:
    """Extract TEL captures from a String column or a tokenized-struct column.

    Returns a struct of capture fields plus a ``complement`` field. Elementwise:
    row *i* depends only on row *i*.
    """
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="extract_expr",
        args=expr,
        kwargs={"model_path": str(model_path), "pattern": pattern, "mode": mode},
        is_elementwise=True,
        use_abs_path=True,
    )


def encode_class_ids(expr: IntoExpr, *, model_path: str | Path) -> pl.Expr:
    """Encode a ``List(String)`` classes column into compact ``List(UInt8)`` ids."""
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="encode_class_ids_expr",
        args=expr,
        kwargs={"model_path": str(model_path)},
        is_elementwise=True,
        use_abs_path=True,
    )
