//! 1,000,000-row benchmark for tokmat-polars.
//!
//! Generates a deterministic corpus of synthetic Canadian-style addresses and
//! measures the two hot paths:
//!   * `tokenize_series`  (String -> tokenized struct)
//!   * `extract_series`   (tokenized struct -> capture struct)
//!
//! Run with:  cargo bench --bench tokmat_bench
//! The synthetic corpus is fixed (seeded) so numbers are comparable across runs.

// Benchmark-only code: relax pedantic lints (truncating row counts to u64/usize
// and prose that reads like code) that do not matter for a bench harness.
#![allow(clippy::cast_possible_truncation, clippy::doc_markdown)]

use std::path::Path;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use polars::prelude::*;
use tokmat_polars::TokmatPolars;

const ROW_COUNT: usize = 1_000_000;
const PATTERN: &str = "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>";

fn fixture_model_path() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/model_1")
        .to_string_lossy()
        .into_owned()
}

/// Small xorshift so the corpus is deterministic without pulling in `rand`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next() as usize) % items.len()]
    }
}

fn build_addresses(count: usize) -> Vec<String> {
    let streets = [
        "MAIN",
        "OAK",
        "KING",
        "QUEEN",
        "MAPLE",
        "ELM",
        "CEDAR",
        "PINE",
        "BIRCH",
        "WELLINGTON",
        "RICHMOND",
        "DUNDAS",
        "BLOOR",
        "YONGE",
        "BAY",
        "FRONT",
    ];
    let types = [
        "ST", "AVE", "RD", "BLVD", "DR", "CRES", "WAY", "LANE", "CRT", "PL",
    ];
    let dirs = ["", " N", " S", " E", " W", " NW", " SE"];

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // Every 17th row is a postal-code-bearing variant to exercise that branch;
        // every 23rd row carries an ATTN prefix to exercise complement handling.
        let civic = (rng.next() % 9999) + 1;
        let street = rng.pick(&streets);
        let stype = rng.pick(&types);
        let dir = rng.pick(&dirs);
        let mut s = String::with_capacity(24);
        if i % 23 == 0 {
            s.push_str("ATTN ");
        }
        s.push_str(&civic.to_string());
        s.push(' ');
        s.push_str(street);
        s.push(' ');
        s.push_str(stype);
        s.push_str(dir);
        if i % 17 == 0 {
            s.push_str(" K1A 0B1");
        }
        out.push(s);
    }
    out
}

fn bench(c: &mut Criterion) {
    let plugin = TokmatPolars::from_model_path(fixture_model_path()).expect("model should load");
    let addresses = build_addresses(ROW_COUNT);
    let input = Series::new("address".into(), addresses);

    // Pre-tokenize once for the extract benchmark so extract is measured in isolation.
    let tokenized = plugin
        .tokenize_series(&input)
        .expect("tokenize for extract setup");

    let mut group = c.benchmark_group("tokmat_1mm");
    group.throughput(Throughput::Elements(ROW_COUNT as u64));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("tokenize_series", |b| {
        b.iter_batched(
            || &input,
            |input| plugin.tokenize_series(input).expect("tokenize"),
            BatchSize::LargeInput,
        );
    });

    group.bench_function("extract_from_string", |b| {
        b.iter_batched(
            || &input,
            |input| {
                plugin
                    .extract_series(input, PATTERN)
                    .expect("extract string")
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("extract_from_tokenized", |b| {
        b.iter_batched(
            || &tokenized,
            |tokenized| {
                plugin
                    .extract_series(tokenized, PATTERN)
                    .expect("extract tokenized")
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
