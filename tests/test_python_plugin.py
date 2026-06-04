from pathlib import Path

import polars as pl
import tokmat_polars as tk
from tokmat_polars import _tokmat_polars


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "tests" / "fixtures" / "model_1"
PATTERN = "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>"

_DISPATCH = {
    "tokenize_expr": tk.tokenize,
    "extract_expr": tk.extract,
    "encode_class_ids_expr": tk.encode_class_ids,
}


def plugin_expr(function_name: str, arg: pl.Expr, **kwargs: object) -> pl.Expr:
    # Route through the public wrapper API, which registers the plugin with
    # is_elementwise=True.
    return _DISPATCH[function_name](arg, **kwargs)


def test_wrapper_exposes_api_and_compiled_module() -> None:
    assert {"tokenize", "extract", "encode_class_ids"} <= set(tk.__all__)
    so_path = Path(_tokmat_polars.__file__)
    assert so_path.suffix in {".so", ".pyd", ".dll"}


def test_tokenize_is_registered_elementwise() -> None:
    # An elementwise plugin participates in streaming and lets Polars resolve the
    # output schema at plan time without executing. Both should hold here.
    lf = pl.LazyFrame({"address": ["123 MAIN ST"]}).select(
        tk.tokenize(pl.col("address"), model_path=str(MODEL_PATH)).alias("tok")
    )
    schema = lf.collect_schema()  # plan-time schema, no execution
    assert isinstance(schema["tok"], pl.Struct)
    # Streaming engine can run the elementwise plugin end to end.
    out = lf.unnest("tok").collect(engine="streaming")
    assert out["tokens"].to_list()[0] == ["123", " ", "MAIN", " ", "ST"]


def test_tokenize_plugin_runs_inside_polars() -> None:
    frame = pl.DataFrame({"address": ["123 MAIN ST", None]})

    tokenized = frame.select(
        plugin_expr(
            "tokenize_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
        ).alias("tokenized")
    ).unnest("tokenized")

    assert tokenized["raw_value"].to_list() == ["123 MAIN ST", None]
    assert tokenized["tokens"].to_list()[0] == ["123", " ", "MAIN", " ", "ST"]
    assert tokenized["classes"].to_list()[0] == ["NUM", " ", "ALPHA", " ", "STREETTYPE"]


def test_tokenize_plugin_can_emit_compact_class_ids() -> None:
    frame = pl.DataFrame({"address": ["123 MAIN ST"]})

    tokenized = frame.select(
        plugin_expr(
            "tokenize_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
            include_raw_value=False,
            include_types=False,
            include_classes=False,
            include_class_ids=True,
        ).alias("tokenized")
    ).unnest("tokenized")

    assert tokenized.columns == ["tokens", "class_ids"]
    assert tokenized["tokens"].to_list()[0] == ["123", " ", "MAIN", " ", "ST"]
    assert all(isinstance(value, int) for value in tokenized["class_ids"].to_list()[0])


def test_tokenize_plugin_can_emit_categorical_lists() -> None:
    frame = pl.DataFrame({"address": ["123 MAIN ST"]})

    tokenized = frame.select(
        plugin_expr(
            "tokenize_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
            include_raw_value=False,
            token_output="categorical",
            type_output="categorical",
            class_output="categorical",
        ).alias("tokenized")
    ).unnest("tokenized")

    assert tokenized.schema["tokens"] == pl.List(pl.Categorical)
    assert tokenized.schema["types"] == pl.List(pl.Categorical)
    assert tokenized.schema["classes"] == pl.List(pl.Categorical)


def test_extract_plugin_accepts_raw_string_series() -> None:
    frame = pl.DataFrame({"address": ["123 MAIN ST"]})

    extracted = frame.select(
        plugin_expr(
            "extract_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
            pattern=PATTERN,
        ).alias("parsed")
    ).unnest("parsed")

    assert extracted.to_dict(as_series=False) == {
        "CIVIC": ["123"],
        "STREET": ["MAIN"],
        "TYPE": ["ST"],
        "complement": [""],
    }


def test_extract_plugin_accepts_tokenized_struct_series() -> None:
    frame = pl.DataFrame({"address": ["123 MAIN ST"]}).with_columns(
        plugin_expr(
            "tokenize_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
        ).alias("tokenized")
    )

    extracted = frame.select(
        plugin_expr(
            "extract_expr",
            pl.col("tokenized"),
            model_path=str(MODEL_PATH),
            pattern=PATTERN,
        ).alias("parsed")
    ).unnest("parsed")

    assert extracted.to_dict(as_series=False) == {
        "CIVIC": ["123"],
        "STREET": ["MAIN"],
        "TYPE": ["ST"],
        "complement": [""],
    }


def test_extract_plugin_accepts_minimal_tokenized_struct_series() -> None:
    frame = pl.DataFrame(
        {
            "tokens": [["123", " ", "MAIN", " ", "ST"]],
            "classes": [["NUM", " ", "ALPHA", " ", "STREETTYPE"]],
        }
    ).with_columns(pl.struct("tokens", "classes").alias("tokenized"))

    extracted = frame.select(
        plugin_expr(
            "extract_expr",
            pl.col("tokenized"),
            model_path=str(MODEL_PATH),
            pattern=PATTERN,
        ).alias("parsed")
    ).unnest("parsed")

    assert extracted.to_dict(as_series=False) == {
        "CIVIC": ["123"],
        "STREET": ["MAIN"],
        "TYPE": ["ST"],
        "complement": [""],
    }


def test_extract_plugin_accepts_compact_class_id_struct_series() -> None:
    frame = pl.DataFrame({"address": ["123 MAIN ST"]}).with_columns(
        plugin_expr(
            "tokenize_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
            include_raw_value=False,
            include_types=False,
            include_classes=False,
            include_class_ids=True,
        ).alias("tokenized")
    )

    extracted = frame.select(
        plugin_expr(
            "extract_expr",
            pl.col("tokenized"),
            model_path=str(MODEL_PATH),
            pattern=PATTERN,
        ).alias("parsed")
    ).unnest("parsed")

    assert extracted.to_dict(as_series=False) == {
        "CIVIC": ["123"],
        "STREET": ["MAIN"],
        "TYPE": ["ST"],
        "complement": [""],
    }


def test_extract_plugin_accepts_categorical_tokenized_struct_series() -> None:
    frame = pl.DataFrame({"address": ["123 MAIN ST"]}).with_columns(
        plugin_expr(
            "tokenize_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
            include_raw_value=False,
            token_output="categorical",
            class_output="categorical",
        ).alias("tokenized")
    )

    extracted = frame.select(
        plugin_expr(
            "extract_expr",
            pl.col("tokenized"),
            model_path=str(MODEL_PATH),
            pattern=PATTERN,
        ).alias("parsed")
    ).unnest("parsed")

    assert extracted.to_dict(as_series=False) == {
        "CIVIC": ["123"],
        "STREET": ["MAIN"],
        "TYPE": ["ST"],
        "complement": [""],
    }


def test_extract_plugin_respects_any_mode_for_raw_strings() -> None:
    frame = pl.DataFrame({"address": ["ATTN 123 MAIN ST"]})

    extracted = frame.select(
        plugin_expr(
            "extract_expr",
            pl.col("address"),
            model_path=str(MODEL_PATH),
            pattern=PATTERN,
            mode="any",
        ).alias("parsed")
    ).unnest("parsed")

    assert extracted.to_dict(as_series=False) == {
        "CIVIC": ["123"],
        "STREET": ["MAIN"],
        "TYPE": ["ST"],
        "complement": ["ATTN "],
    }
