# tokmat-polars / tokmat performance work — findings & timings

_Internal engineering report. Date: 2026-06-03._

This documents a performance pass across `tokmat-polars` and its `tokmat`
dependency: the optimizations applied, the benchmark methodology, measured
timings (with their caveats), two regressions found and diagnosed, and the
root-cause of why parallel **extraction** does not scale.

---

## 1. TL;DR

- **Tokenization is ~8× faster** (5.22 s → 0.66 s on 1M rows). Drivers: parallel
  tokenize, `raw_value` passthrough, single-pass classifier, and removing a
  per-row global `RwLock` read that was contending under parallel tokenize (a
  side benefit of the per-model word-definition work — see §8).
- **Biggest find — extraction was paying a per-row PCRE2 JIT-stack `mmap`.** The
  comparator regexes set a 5 MB `max_jit_stack_size`, and `pcre2`'s `captures()`
  allocates a fresh `MatchData` (with that JIT stack) **per row** → `mmap`/`munmap`
  of 5 MB every row. It dominated **86% of even serial extract**, and serialized
  on the kernel `mmap` lock under threads. Removing it: `extract_from_tokenized`
  **7.0 s → 0.98 s serial (7×)** and **22 s → 0.69 s parallel (now scales)**;
  `extract_from_string` 14.5 s → 3.76 s. See §7. (A proper fix that keeps
  backtracking headroom = reuse the `MatchData`; decision flagged.)
- **Parallel extraction is opt-in** (`TOKMAT_ENABLE_RAYON`) — counter-productive
  *until* the JIT-stack fix lands, after which it beats serial and could be
  default. The earlier "Extractor `Mutex` cache contention" theory was **tested
  and disproven** (per-chunk Extractors changed nothing); so were allocator and
  malloc-arena theories. The model loaded from `.param`/`.param2` is shared
  read-only via `Arc` and was never the bottleneck. §7.2 has the full trail.
- **Regression fixed:** an in-progress `tokmat` `word_definition` refactor added
  per-row `String` allocs + `RwLock` reads (+20–38% on extract); fixed with
  pre-rendered `Arc<str>`.
- A **French-Canadian / Unicode** classifier idea was rejected: the enforced
  wanParser parity corpus requires accented unknown tokens to keep their raw
  type, not become `ALPHA`.

---

## 2. Benchmark methodology

- Harness: `benches/tokmat_bench.rs` (criterion), 1,000,000 synthetic
  Canadian-style address rows, deterministic xorshift corpus (varied civic
  numbers + street names/types, periodic postal codes and `ATTN` prefixes).
- Three measured paths:
  - `tokenize_series` — String → tokenized struct.
  - `extract_from_string` — String → capture struct (tokenizes internally).
  - `extract_from_tokenized` — pre-tokenized struct → capture struct.
- Fixture model: `tests/fixtures/model_1` (has `POSTALCODE`, `NUM`, `ALPHA`,
  `STREETTYPE`, `PROV`, …).
- Hardware: 16 cores.

### Thermal caveat (important)

Over a long benchmarking session the CPU drifted measurably slower: e.g. the
**same** `extract_from_string` code read 11.4 s early and 14.3 s late;
`extract_from_tokenized` baseline drifted 7.4 s → 10.6 s. Tokenize (single-core)
was thermally stable. **Consequence:** large deltas are trustworthy; small ones
(≤ ~5%) are within the drift floor and should be treated as directional only.
The final numbers below come from a **back-to-back A/B** (both arms measured in
the same thermal state) to remove this confound.

---

## 3. Timings

### 3.1 Clean A/B — `HEAD` (committed) vs final working tree

Both arms built and measured back-to-back, same thermal state. Baseline =
committed `HEAD` + crates.io `tokmat 0.2.0` + default release profile. Final =
all changes + local `tokmat` + thin-LTO profile.

| Metric | Baseline (HEAD) | Final | Δ |
|---|---|---|---|
| `tokenize_series` | 5.184 s | **1.633 s** | **−68.5%** |
| `extract_from_string` | 15.27 s | 14.56 s | −4.6% |
| `extract_from_tokenized` | 10.61 s | 7.15 s | −32.6% |

### 3.2 Honest attribution to *this* session

The `HEAD` baseline lacks a **pre-existing uncommitted optimization** that was
already in `src/lib.rs` when this work started (session-start file ≈ 2212 lines;
`HEAD` is 2147). Evidence: session-start `extract_from_tokenized` baseline was
**7.41 s**, but `HEAD` baseline is **10.6 s**, while tokenize baselines match
(~5.2 s) — so ~30% of the extract A/B delta belongs to that prior change, not
this session.

Corrected per-session attribution:

| Metric | This session's real impact |
|---|---|
| `tokenize_series` | **~−69%** (parallel tokenize + `raw_value` + classifier) |
| `extract_from_tokenized` | ~−3–4% (builders + thin LTO); rest is pre-existing |
| `extract_from_string` | ~−4% (single-pass `tokmat` classifier) |

### 3.3 Per-change progression (with thermal caveat)

| Step | tokenize | extract_str | extract_tok | Notes |
|---|---|---|---|---|
| Baseline (crates.io tokmat, no profile) | 5.224 | 12.49 | 7.409 | session start, cool |
| + release profile (fat LTO) | 5.117 | 11.77 | 6.945 | −2% / −5.8% / −6.3% |
| + `raw_value` passthrough | 4.307 | 11.55 | 6.806 | **−16% tokenize** |
| + single-pass classifier (tokmat-polars) | 3.940 | — | — | −8.5% tokenize |
| swap to local `tokmat` WIP | 4.009 | 13.89 | 9.467 | **regression +20/+38%** |
| + `tokmat` `word_definition` fix | 4.029 | 11.87 | 6.972 | recovered (−14.5/−26%) |
| + single-pass classifier (tokmat) | 4.051 | 11.41 | 6.901 | −3.9% extract_str |
| + parallelism (tokenize+extract) | 1.68 | 14.1 | 20.3 | **extract +195% (contention)** |
| + split gate (extract serial again) | 1.67 | 14.3 | 7.19 | tokenize keeps −58% |
| + builder extract output | — | 14.5 | 7.44 | memory win; thermal drift |
| + #4 atomics + #5 hoist compile | — | 14.0 | 7.0 | per-row locks removed; marginal |
| **+ drop per-row JIT-stack (§7)** | — | **3.76** | **0.98** (par **0.69**) | **extract 4–7× faster; parallel scales** |

(Switched fat→thin LTO mid-way for faster iteration; the shipped wheel can use
fat LTO in CI via `CARGO_PROFILE_RELEASE_LTO=fat`. The JIT-stack row is the
single biggest extract win and is the open decision in §7.3/§8.)

### 3.4 Extract after the JIT-stack fix (the headline)

| Metric | before (5 MB JIT stack/row) | after | speedup |
|---|---|---|---|
| `extract_from_tokenized` serial | ~7.0 s | **0.98 s** | **7.1×** |
| `extract_from_tokenized` parallel (16t) | ~22 s | **0.69 s** | **~32×** (now scales) |
| `extract_from_string` serial | 14.5 s | **3.76 s** | 3.9× |

### 3.5 Final state (after the per-model word-definition work)

Measured together with a single `cargo bench` (defaults: tokenize parallel,
extract serial), on a cooled machine. **No regression**, and tokenize improved
again (see §8 — removing the per-row global `RwLock` read fixed a parallel
contention point):

| Metric | prior best | final | vs original baseline (§3.3) |
|---|---|---|---|
| `tokenize_series` | 1.63 s | **0.66 s** | 5.22 s → 0.66 s (**~8×**) |
| `extract_from_string` | 3.76 s | **3.80 s** | ~12.5 s → 3.80 s (**~3.3×**) |
| `extract_from_tokenized` | 0.98 s | **0.99 s** | ~7.4 s → 0.99 s (**~7.5×**) |
| `extract_from_tokenized_lowcard` | — | 3.40 s | (new) |

(Part of the tokenize gain vs the §3.1 A/B is thermal recovery — the A/B ran on a
heavily-loaded machine — but the contention removal is a genuine structural win.)

---

## 4. Optimizations applied

### tokmat-polars
1. **Release profile** — `lto`, `codegen-units = 1`, `strip`. `panic = "unwind"`
   kept so a Rust panic surfaces as a Python exception across PyO3.
2. **`raw_value` passthrough** — the `raw_value` struct field is byte-for-byte
   the input column, so reuse the input `Series` (Arc clone + rename) instead of
   rebuilding it cell-by-cell and re-copying into Arrow. Biggest single tokenize
   win (−16%): removes 1M `String` allocations + a full column copy.
3. **Single-pass `classify_token_ref`** — one byte pass computing all
   character-class predicates + a non-allocating `POSTALCODE` check, replacing
   ~10 `chars()` scans and 2 allocations per token. ASCII-gated (see §6).
4. **Parallel tokenize (default-on)** — slice the input into row ranges, run the
   serial path per range under rayon, append the struct chunks. Validated by a
   `parallel == serial` equality test. Opt-out via `TOKMAT_DISABLE_RAYON`.
5. **Builder-based extract output** — serial extract streams each capture +
   complement straight into `StringChunkedBuilder`, moving values out of
   `ParseOutput.fields` (`remove`, not `clone`). Removes the intermediate
   `Vec<Option<String>>` duplicate of the whole output column-set + one copy
   (memory win; speed-neutral).
6. **Split parallel gate** — tokenize parallel by default; **extract parallel is
   opt-in** (`TOKMAT_ENABLE_RAYON`) because it regresses (see §7).

### tokmat
7. **`word_definition` regression fix** — see §5.
8. **Single-pass `get_token_fast_classifier`** — same one-pass rewrite as (3);
   also benefits `extract_from_string` (~−4%) since it runs per token there.
9. **Flaky-test fix** — a `#[cfg(test)] RwLock` (`WORD_DEF_TEST_LOCK`) serializes
   the test that mutates the process-global word definition against tests that
   read the default boundary (was racing under the parallel test runner).
10. **Atomic `execution_counters`** (§7.4 #4) — was `Mutex<ExecutionCounters>`
    locked several times per row; now lock-free `AtomicU64`s.
11. **PCRE2 JIT-stack change** (§7.1/§7.3) — the dominant extract cost. Currently
    the *quick* form (drop `max_jit_stack_size`) is applied as the confirmation;
    the recommended form is `MatchData` reuse. **Decision needed before shipping.**

(The pattern-compile hoist (§7.4 #5) and per-chunk Extractors (§7.4 #1) live in
`tokmat-polars`, not `tokmat`.)

---

## 5. Regression #1 — `tokmat` `word_definition` per-row allocations

The in-progress `word_definition.rs` refactor (configurable word character
class) replaced compile-time `const` regex strings and a `static LazyLock`
boundary regex with **runtime accessors behind an `RwLock`**, whose derived-form
methods `format!` a fresh `String` per call (`word_regex()`, `boundary()`,
`char_class()`), and `word_definition()` clones the definition under the lock.

Why it showed up per-row: `ObjectPlanCacheKey` includes `captured_groups` (the
literal captured token values). High-cardinality address data (random civic
numbers) misses the object-plan cache almost every row, so the plan-build path —
which calls those accessors — runs per row. The accessors went from zero-cost
`const` reads to several small heap allocations + lock reads per row → **+20%
(`extract_from_string`) / +38% (`extract_from_tokenized`)**.

**Fix:** pre-render the derived forms as `Arc<str>` in `WordState` (built once at
configure time) and return cheap `Arc` clones; update the ~8 call sites in
`tel.rs`/`extractor.rs`. Recovered extraction to ~published-0.2.0 levels.

---

## 6. French-Canadian / Unicode classifier (rejected)

Idea: make `ALPHA`/`ALPHA_EXTENDED`/`ALPHA_NUM*` use Unicode `is_alphabetic` so
accented tokens (`ALLÉE`, `RIVIÈRE`) classify via the fast path. **Rejected:**
the enforced `test_tokenizer_parity` corpus expects accented unknown tokens to
keep their raw type (input `7078 Saint-Laurent Blvd Étg 630` → `Étg` stays
`"ÉTG"`, not `"ALPHA"`). The ASCII gate is intentional. French-Canadian is
handled by the **token-class dictionary** (accented known tokens → class, e.g.
`ALLÉE` → `STREETTYPE`) and **upstream `strip_accents` normalization** (shipped
in `tokmat::normalize`, applied by the caller), not by the classifier. The
single-pass rewrite was kept (ASCII-gated, parity-preserving, faster).

---

## 7. Regression #2 — why EXTRACTION is slow (the real root cause)

**Symptom:** enabling the existing parallel extract path made
`extract_from_tokenized` **~3× slower** (7 s → 22 s) with only ~330% CPU
(≈3 of 16 cores) — i.e. threads were *blocked*, not computing.

This took a multi-hypothesis investigation. The first instinct (the `Extractor`'s
`Mutex`-guarded LRU caches) turned out to be **wrong**, and the benchmark proved
it. The journey is worth recording because every dead end eliminated a plausible
cause.

### 7.1 The actual root cause (confirmed)

`tokmat`'s comparator regexes (`tel.rs`/`extractor.rs` `compile_pcre2_regex`) set
`.max_jit_stack_size(Some(FANCY_REGEX_BACKTRACK_LIMIT))` — a **5 MB** PCRE2 JIT
stack. In the `pcre2` crate, the JIT stack lives **inside `MatchData`**
(`ffi.rs`: `MatchData::new` → `pcre2_jit_stack_create` when a max size is set,
freed on drop). And `Regex::captures()` allocates a **fresh `MatchData` per
call** (it does *not* use the match-data reuse pool that `is_match`/`find` use).

`tokmat`'s `run_pcre2_captures` calls `regex.captures(...)` **per row**. So every
extracted row did:

```
pcre2_jit_stack_create(5 MB)   // mmap of executable memory
... match ...
pcre2_jit_stack_free()         // munmap
```

`mmap`/`munmap` of a 5 MB region **per row**:
- is expensive even single-threaded — it dominated **86% of serial extract time**
  (the profiler's `class_regex` stage was mostly this, not matching);
- serializes under threads on the **kernel's per-process address-space lock**
  (`mmap_lock`), which is why parallel collapsed and CPU sat at ~330%.

Tokenize's regexes leave `max_jit_stack_size = None` (no per-call JIT stack, and
they use the pooled `find`/`is_match` path), so tokenize has no per-row mmap and
scales fine (−58%).

**Confirmation (dropping `max_jit_stack_size`):**

| `extract_from_tokenized` | with 5 MB JIT stack | without |
|---|---|---|
| serial | ~7.0 s | **0.98 s** (7.1× faster) |
| parallel (16 threads) | ~22 s | **0.69 s** (now scales, ~32× faster) |

`extract_from_string` serial: **14.5 s → 3.76 s** (3.9×; it also tokenizes, so a
smaller share is extract). All `tokmat` (26 + parity) and `tokmat-polars` (12)
tests still pass.

### 7.2 Hypotheses tested and disproven (the debugging trail)

Each was implemented/measured, not just reasoned about:

| # | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | `Extractor` LRU-cache `Mutex` contention | **per-chunk Extractors** (each rayon chunk gets its own caches) | **No change** (21.8 → 21.9 s). Disproves the cache-lock theory. |
| H2 | Per-row plan **rebuild** cost (high-cardinality misses) | **low-cardinality corpus** (identical rows → plan cache hits) | Still ~2.7× slower in parallel. Rules out the build path. |
| H3 | Rust allocator contention | **mimalloc** as `#[global_allocator]` | No change. (Clue: it doesn't intercept libpcre2's **C** allocations.) |
| H4 | C `malloc` arena contention | **tcmalloc via `LD_PRELOAD`** (does intercept C malloc) | No change → not malloc-arena; pointed at **mmap**, not malloc. |
| H5 | Generic PCRE2 match / word-definition `RwLock` | tokenize uses both heavily | Tokenize **scales −58%** → not these. The difference is `captures` (extract) vs `find`/`is_match` (tokenize). |

Two diagnostics nailed it:
- **Thread scaling:** `RAYON_NUM_THREADS` = 1 → **7.0 s** (== serial; per-chunk
  machinery adds zero overhead), 4 → 21 s, 16 → 22 s. A hard serialization point
  that saturates by 4 threads.
- **CPU sampling:** ~330% CPU with 17 threads → workers blocked in the kernel
  (consistent with `mmap_lock`, not user-space spinning or bandwidth).

Reading the `pcre2` source then revealed `captures()` → fresh `MatchData` →
`pcre2_jit_stack_create` (mmap) per call.

### 7.3 Fixes

- **Proper fix (recommended):** keep the JIT-stack capacity but **reuse the
  `MatchData`** instead of allocating per row — use `captures_read(&mut locs, …)`
  with a `CaptureLocations` created once and reused across rows (one mmap per
  thread, not per row). Preserves deep-backtracking headroom *and* removes the
  mmap storm.
- **Quick fix (currently applied as the confirmation, flagged in `tel.rs`):**
  drop `max_jit_stack_size`, letting PCRE2 use its default JIT stack. Simplest and
  gives the full speedup, but reduces backtracking headroom — safe only if the
  comparator patterns never need deep JIT backtracking (parity corpus passes;
  unverified for all real models). **Decision needed before shipping.**
- **Alternative:** disable JIT for the comparator (`jit_if_available(false)`) — no
  JIT stack at all; slightly slower per match but no per-row allocation.

### 7.4 Other extract work done along the way (kept; smaller wins)

These were implemented during the investigation. They remove real per-row locks
and help, but are dwarfed by the JIT-stack fix:

- **#4 — atomic `execution_counters`.** Was `Mutex<ExecutionCounters>` locked
  several times per row for stats; now lock-free atomics.
- **#5 — hoisted pattern compile.** The compiled TEL pattern is identical for all
  rows of a call; `tokmat-polars` now compiles once (`compile_extract_pattern`)
  and calls `parse_compiled_tokens_with_views`, removing the per-row
  `compiled_pattern_cache.lock()`.
- **#1 — per-chunk Extractors** (kept; harmless, zero overhead at 1 thread). Did
  not fix the regression (H1) but is the right structure if the JIT-stack fix
  lands and the remaining cache locks ever matter.
- **#3 — concurrent cache: not done** (moot — H1 showed cache locking is not the
  bottleneck).

**Decision (current):** parallel extract is gated behind `TOKMAT_ENABLE_RAYON`.
Once the JIT-stack fix lands it should be revisited — parallel extract then
*beats* serial (0.69 s vs 0.98 s) and could become the default.

---

## 8. Word definition — per-model isolation + exposure

The configurable word character class (`WORDDEFINITION.param`) lived behind a
**process-global** `RwLock<WordState>` in `tokmat`, and `TokenModel::load`
**wrote that global as a side effect**. With `tokmat-polars` caching multiple
models, the global became "whatever model loaded last" — so model A's
tokenization/extraction could silently use model B's word boundary. (This
surfaced concretely: a new test that loads a custom-word-def model made the
`model_1` extract tests fail in parallel — the exact hazard.)

### What was done

- **`TokenModel` retains its `WordDefinition` + compiled `word_boundary`**
  (`token_model.rs`), and **`load()` no longer mutates the global** — loading a
  model has no global side effect anymore.
- **Tokenization is fully per-model.** New `split_input_tokens_with(input,
  &boundary)`; `tokenize_with_model` and `tokmat-polars`'s tokenize paths use
  `model.word_boundary()`. No global read, no race. Proven by
  `per_model_word_definition_changes_tokenization` (custom def keeps `192.168`
  one token; default splits it; both deterministic regardless of load order).
  - **Bonus: this was also a ~2.5× parallel-tokenize speedup** (1.63 s → 0.66 s).
    The old path called `tokenizer_boundary()` — a global `RwLock` read + `Arc`
    clone — **per row**; under 16 threads every row contended on that one global
    lock's reader count (same contention class as the extract investigation).
    Passing the model's boundary explicitly removes the per-row global read, so
    parallel tokenize scales. A regression check (the user's ask) found no
    regression anywhere and this improvement.
- **Extraction re-applies the model's def per call.** The TEL compiler and
  object-plan builder still read the global, so `extract_series_with_context`
  calls `configure_word_definition(context.word_definition)` at the top. This
  makes sequential multi-model extraction correct.
- **Exposure (`#2`/`#3`):**
  - Rust: `TokmatPolars::word_definition() -> &str`; `pub use … WordDefinition`.
  - Python (registered in the `pymodule`): `model_word_definition(path)`,
    `get_word_definition()`, `set_word_definition(chars)`. Verified by importing
    the built `.so`: `model_word_definition(model_worddef)` → `\w\-'.`,
    `set/get` round-trips.
- **Fixture:** `tests/fixtures/model_worddef/` (adds `.` as a word char).

### Residual (documented limitation)

Truly **concurrent** extraction of *different* models in one process can still
race the global during the TEL/object-plan phase. Full isolation needs the word
definition threaded through `CompiledPattern::compile` and the `Extractor`
(making the comparator/class-plan build take a `&WordDefinition` instead of
reading the global) — a deeper `tokmat` core-API change. Tokenization is already
fully isolated; extraction is correct for the normal one-model-per-call usage.

---

## 9. Caveats / before-merge checklist

- **PCRE2 JIT-stack change in `tel.rs` is the open decision (§7.3).** As applied
  it *drops* `max_jit_stack_size` (the quick confirmation). Decide between: (a)
  keep the drop (simplest, full speedup, reduced backtracking headroom), or (b)
  the recommended `MatchData`-reuse fix (keeps headroom, same speedup, more code).
  Re-check whether `extractor.rs` should get the same treatment.
- **`[patch.crates-io] tokmat = { path = "../tokmat" }`** in
  `tokmat-polars/Cargo.toml` is **dev-only** — replace with a published `tokmat`
  version bump before release.
- **Bench-only experiment artifacts** in `benches/tokmat_bench.rs` /
  `[dev-dependencies]`: `mimalloc` as `#[global_allocator]` and the
  `extract_from_tokenized_lowcard` benchmark were investigation tools. mimalloc
  did not help the target (it doesn't touch libpcre2's C allocations); keep or
  drop as preferred — they don't affect the shipped library.
- The `tokmat` working tree **mixes pre-existing WIP with these fixes.** This
  session's `tokmat` changes: `word_definition.rs` (`Arc<str>` caching + test
  lock), `tokenizer.rs` (single-pass classifier + parity test + test lock),
  `extractor.rs` (atomic counters + word-def call sites), `tel.rs` (word-def call
  sites + JIT-stack change). `token_model.rs` (+48) is pre-existing WIP.
- `lib.rs` working tree also contains a **pre-existing uncommitted optimization**
  that predates this session (see §3.2) — relevant when attributing the extract
  numbers.
- Tests green in both repos: `tokmat-polars` 12, `tokmat` 26+ (incl. parity),
  stable under the parallel test runner.

---

## 10. File-by-file change summary

**tokmat-polars**
- `Cargo.toml` — release profile (thin LTO); `[dev-dependencies]` criterion +
  mimalloc (bench-only); `[[bench]]`; dev-only `[patch.crates-io]`.
- `src/lib.rs` — `raw_value` passthrough; single-pass `classify_token_ref`;
  parallel tokenize (`tokenize_series_parallel` / `_chunked`); split parallel
  gates (`should_parallelize_tokenize` / `_extract`); builder-based extract
  output (`ExtractColumnBuilders`); hoisted pattern compile
  (`compile_extract_pattern` + `parse_compiled_tokens_with_views`); per-chunk
  Extractors (`fresh_chunk_extractor`); per-model tokenize via
  `split_input_tokens_with(model.word_boundary())`; word-def on `ModelContext` +
  re-apply in `extract_series_with_context`; `TokmatPolars::word_definition()`;
  `pub use WordDefinition`; pymodule funcs (`model_word_definition`,
  `get_word_definition`, `set_word_definition`); tests.
- `benches/tokmat_bench.rs` — new 1M-row criterion harness; `mimalloc`
  global allocator (experiment); `extract_from_tokenized_lowcard` (experiment).
- `tests/fixtures/model_worddef/` — new fixture with a custom `WORDDEFINITION.param`.

**tokmat**
- `src/word_definition.rs` — `Arc<str>` pre-rendered accessors; `WORD_DEF_TEST_LOCK`;
  `WordDefinition::compiled_boundary()`.
- `src/tokenizer.rs` — single-pass `get_token_fast_classifier`; parity/FR-CA test;
  test lock; `ascii_is_whitespace` helper; `split_input_tokens_with`;
  `tokenize_with_model` uses the model boundary.
- `src/token_model.rs` — `TokenModel` retains `word_definition` + `word_boundary`
  with accessors; `load()` no longer writes the process-global.
- `src/extractor.rs` — atomic `execution_counters` (lock-free per-row stats);
  cached `Arc<str>` word-definition call sites.
- `src/tel.rs` — cached `Arc<str>` word-definition call sites; **JIT-stack change**
  (`compile_pcre2_regex`, §7.3 — open decision).
