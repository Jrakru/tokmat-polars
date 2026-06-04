# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [0.3.0] - 2026-06-04

### Added

- Python wrapper API — `tokmat_polars.tokenize`, `extract`, `encode_class_ids` —
  that registers the plugin expressions with `is_elementwise=True`, so the Polars
  engine can stream them and parallelize across chunks on its own pool. The
  compiled extension is now the private submodule `tokmat_polars._tokmat_polars`
  (mixed maturin layout).
- Per-model word definition: loads `WORDDEFINITION.param`, with the Python
  functions `model_word_definition`, `get_word_definition`, `set_word_definition`.
- `benchmarks/bench_polars_engine.py` — benchmarks the plugin through the Polars
  engine (in-memory vs streaming, `is_elementwise` on vs off).

### Changed

- Internal `rayon` parallelism is now an eager, top-level-only fallback: it is
  skipped when already running on a `rayon` worker (the Polars engine pool), so it
  never nests or oversubscribes. For large data, prefer
  `collect(engine="streaming")` — extraction is ~7x faster and memory-bounded.
- Depends on the published `tokmat = "0.3.0"`.

### Performance

- `raw_value` passthrough (reuse the input column), single-pass token classifier,
  builder-based extract output (no intermediate `Vec`), hoisted per-call pattern
  compile, per-chunk extractors for the opt-in parallel path.

### Fixed

- Extraction from a categorical/enum tokenized struct (`expected String, got cat`):
  token/class columns are cast to `List(String)` once before the `&str`-view path
  (`ensure_string_list_field`).

### Security

- Release wheels set `PCRE2_SYS_STATIC=1`, statically linking the patched vendored
  PCRE2 (>= 10.46) instead of a possibly-vulnerable system `libpcre2-8`
  (CVE-2025-58050). This also makes the wheels self-contained.

### Verified

- `cargo fmt --all --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`
- `cargo publish --dry-run`
- `pytest -q tests/test_python_plugin.py`

## Earlier releases

0.1.x and 0.2.x history is available in the Git log and the GitHub release tags.
