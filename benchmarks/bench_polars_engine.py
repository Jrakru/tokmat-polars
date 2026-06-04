"""Benchmark tokmat-polars through the *Polars engine* (the real path).

Unlike the Rust criterion bench (which calls the Rust API directly), this drives
the plugin the way users do: as a Polars expression in a LazyFrame, collected
with both the in-memory and the streaming engines, and with the plugin registered
both `is_elementwise=True` (the wrapper default) and `is_elementwise=False` (the
naive default) so the difference is visible.

Run:  .venv/bin/python benchmarks/bench_polars_engine.py
"""

from __future__ import annotations

import random
import resource
import time
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

import tokmat_polars as tk
from tokmat_polars import _tokmat_polars

ROWS = 1_000_000
MODEL = str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "model_1")
PATTERN = "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>"
_PLUGIN_PATH = Path(_tokmat_polars.__file__)


def make_frame() -> pl.DataFrame:
    rng = random.Random(0xC0FFEE)
    streets = ["MAIN", "OAK", "KING", "QUEEN", "MAPLE", "ELM", "CEDAR", "PINE",
               "BLOOR", "YONGE", "BAY", "FRONT", "DUNDAS", "RICHMOND"]
    types = ["ST", "AVE", "RD", "BLVD", "DR", "CRES", "WAY", "LANE", "CRT", "PL"]
    dirs = ["", " N", " S", " E", " W", " NW", " SE"]
    rows = [
        f"{rng.randint(1, 9999)} {rng.choice(streets)} {rng.choice(types)}{rng.choice(dirs)}"
        for _ in range(ROWS)
    ]
    return pl.DataFrame({"address": rows})


def raw(function_name: str, expr: pl.Expr, *, is_elementwise: bool, **kwargs):
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name=function_name,
        args=expr,
        kwargs=kwargs,
        is_elementwise=is_elementwise,
        use_abs_path=True,
    )


def time_it(label: str, build_expr, *, engine: str) -> None:
    """Run a single collect and report wall time + peak RSS delta."""
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    try:
        lf = pl.LazyFrame({"address": FRAME["address"]}).select(build_expr())
        n = lf.collect(engine=engine).height
    except Exception as exc:  # noqa: BLE001 - report engine fallbacks/failures
        print(f"  {label:38s} {engine:10s}  -> ERROR: {type(exc).__name__}: {exc}")
        return
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = max(0, (rss1 - rss0)) / 1024
    print(f"  {label:38s} {engine:10s}  {dt:7.3f} s   {n:>9,} rows   +{peak_mb:6.0f} MB peak")


FRAME = make_frame()


def main() -> None:
    print(f"tokmat-polars Polars-engine benchmark — {ROWS:,} rows\n")
    print(f"{'operation / registration':40s} {'engine':10s}  {'time':>9s}   {'rows':>9s}   peak")
    print("-" * 90)

    cases = [
        ("tokenize  is_elementwise=True",
         lambda: tk.tokenize(pl.col("address"), model_path=MODEL).alias("o")),
        ("tokenize  is_elementwise=False",
         lambda: raw("tokenize_expr", pl.col("address"), is_elementwise=False,
                     model_path=MODEL).alias("o")),
        ("extract   is_elementwise=True",
         lambda: tk.extract(pl.col("address"), model_path=MODEL, pattern=PATTERN).alias("o")),
        ("extract   is_elementwise=False",
         lambda: raw("extract_expr", pl.col("address"), is_elementwise=False,
                     model_path=MODEL, pattern=PATTERN).alias("o")),
    ]

    for label, build in cases:
        for engine in ("in-memory", "streaming"):
            time_it(label, build, engine=engine)
        print()


if __name__ == "__main__":
    main()
