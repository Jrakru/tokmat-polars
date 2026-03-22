from pathlib import Path

import polars as pl
import tokmat_polars


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "tests" / "fixtures" / "model_1"
PLUGIN_PATH = Path(tokmat_polars.__file__).resolve().parent
PATTERN = "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>"


def plugin_expr(function_name: str, arg: pl.Expr, **kwargs: object) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name=function_name,
        args=arg,
        kwargs=kwargs,
        use_abs_path=True,
    )


def test_imported_extension_exposes_binary_module() -> None:
    assert tokmat_polars.__file__
    assert PLUGIN_PATH.is_dir()
    assert any(path.suffix in {".so", ".pyd", ".dll"} for path in PLUGIN_PATH.iterdir())


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
