//! Channel B for Engram's vector recall (docs/SELF-EVOLUTION.md, Part 25). Two indexes
//! over one fixed, deterministic workload: the **exact** `VectorIndex` (the O(N) oracle)
//! and the **approximate** `IvfIndex` (scans `PROBE` of `ANCHORS` buckets). Reported:
//! `exact_ns_per_op` / `ivf_ns_per_op` (speed, lower = fitter — the win an ANN index
//! exists for) and `recall` (the fraction of the exact top-k the IVF also returns — the
//! *quality* traded for that speed, a channel-B objective like the Bloom filter's FP
//! rate, Part 18). `top` is the behavioral pin: a nearest neighbour planted at id 0
//! (identical to the query) must rank first in the exact search, every run. Run with
//! `cargo bench --bench engram` or `scripts/fitness-engram.sh`.

use engram::vector::{IvfIndex, VectorIndex};
use std::time::Instant;

const DIMS: usize = 64;
const N: u64 = 5000;
const K: usize = 10;
const ANCHORS: u64 = 64; // ~sqrt(N) buckets
const PROBE: usize = 8; // buckets scanned per approximate search
const WARMUP: u32 = 5;
const RUNS: u32 = 9; // timed samples; the median resists interference spikes
const REPS: u32 = 20;

/// A tiny deterministic splitmix64 → `f32` in `[0, 1)`, so the workload is fully
/// reproducible without an rng dependency.
fn splitmix(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / (1u32 << 24) as f32
}

fn vector(state: &mut u64) -> Vec<f32> {
    (0..DIMS).map(|_| splitmix(state)).collect()
}

fn median(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn time_search(mut run: impl FnMut()) -> u64 {
    for _ in 0..WARMUP {
        run();
    }
    let mut samples = Vec::new();
    for _ in 0..RUNS {
        let t = Instant::now();
        for _ in 0..REPS {
            run();
        }
        samples.push(t.elapsed().as_nanos() as u64 / REPS as u64);
    }
    median(samples)
}

fn main() {
    // Deterministic corpus: id 0 is the exact nearest (a copy of the query).
    let mut qs = 1u64;
    let query = vector(&mut qs);
    let mut anchor_state = 7u64;
    let anchors: Vec<Vec<f32>> = (0..ANCHORS).map(|_| vector(&mut anchor_state)).collect();

    let mut exact = VectorIndex::new(DIMS);
    let mut ivf = IvfIndex::new(DIMS, anchors);
    exact.insert(0, query.clone()).unwrap();
    ivf.insert(0, query.clone()).unwrap();
    let mut state = 42u64;
    for id in 1..N {
        let v = vector(&mut state);
        exact.insert(id, v.clone()).unwrap();
        ivf.insert(id, v).unwrap();
    }

    let exact_ns = time_search(|| {
        exact.search(&query, K);
    });
    let ivf_ns = time_search(|| {
        ivf.search(&query, K, PROBE);
    });

    // Quality: how much of the exact top-k does the approximate search recover?
    let exact_top: Vec<u64> = exact.search(&query, K).iter().map(|h| h.id).collect();
    let ivf_top: Vec<u64> = ivf.search(&query, K, PROBE).iter().map(|h| h.id).collect();
    let overlap = exact_top.iter().filter(|id| ivf_top.contains(id)).count();
    let recall = overlap as f64 / K as f64;
    let top = *exact_top.first().unwrap();

    let exact_ns_per_op = exact_ns as f64 / N as f64;
    let ivf_ns_per_op = ivf_ns as f64 / N as f64;
    println!(
        "{{\"benchmark\":\"engram\",\"dims\":{DIMS},\"n\":{N},\"k\":{K},\"anchors\":{ANCHORS},\"probe\":{PROBE},\"top\":{top},\"recall\":{recall:.3},\"exact_ns_per_op\":{exact_ns_per_op:.3},\"ivf_ns_per_op\":{ivf_ns_per_op:.3}}}"
    );
}
