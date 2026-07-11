//! Channel B for Mnema's vector recall (docs/SELF-EVOLUTION.md, Part 25). Two indexes
//! over one fixed, deterministic workload: the **exact** `VectorIndex` (the O(N) oracle)
//! and the **approximate** `IvfIndex` (scans `PROBE` of `ANCHORS` buckets). Reported:
//! `exact_ns_per_op` / `ivf_ns_per_op` (speed, lower = fitter — the win an ANN index
//! exists for) and `recall` (the fraction of the exact top-k the IVF also returns — the
//! *quality* traded for that speed, a channel-B objective like the Bloom filter's FP
//! rate, Part 18). `top` is the behavioral pin: a nearest neighbour planted at id 0
//! (identical to the query) must rank first in the exact search, every run. Run with
//! `cargo bench --bench mnema` or `scripts/fitness-mnema.sh`.

use mnema::vector::{IvfIndex, VectorIndex, kmeans_anchors};
use std::time::Instant;

const DIMS: usize = 64;
const N: u64 = 5000;
const K: usize = 10;
const ANCHORS: usize = 64; // ~sqrt(N) buckets
const PROBE: usize = 8; // buckets scanned per approximate search
const CLUSTERS: usize = 64; // topical clusters in the realistic workload
const SPREAD: f32 = 0.10; // per-dimension jitter around a cluster centre
const KMEANS_ITERS: usize = 10;
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

/// A vector jittered around `centre` by ±SPREAD/2 per dimension — a point inside a cluster.
fn near(centre: &[f32], state: &mut u64) -> Vec<f32> {
    centre
        .iter()
        .map(|&x| x + SPREAD * (splitmix(state) - 0.5))
        .collect()
}

/// The exact top-K ids for `query` over `corpus` (the oracle every recall is measured against).
fn exact_top(corpus: &[(u64, Vec<f32>)], query: &[f32]) -> Vec<u64> {
    let mut idx = VectorIndex::new(DIMS);
    for (id, v) in corpus {
        idx.insert(*id, v.clone()).unwrap();
    }
    idx.search(query, K).iter().map(|h| h.id).collect()
}

/// Fraction of the exact top-K that an IVF built on `anchors` recovers at PROBE buckets.
fn recall(
    corpus: &[(u64, Vec<f32>)],
    query: &[f32],
    anchors: Vec<Vec<f32>>,
    oracle: &[u64],
) -> f64 {
    let mut ivf = IvfIndex::new(DIMS, anchors);
    for (id, v) in corpus {
        ivf.insert(*id, v.clone()).unwrap();
    }
    let got: Vec<u64> = ivf.search(query, K, PROBE).iter().map(|h| h.id).collect();
    oracle.iter().filter(|id| got.contains(id)).count() as f64 / K as f64
}

fn main() {
    // --- Worst case: structureless uniform-random vectors (no clusters to find). ---
    let mut qs = 1u64;
    let rq = vector(&mut qs);
    let mut random: Vec<(u64, Vec<f32>)> = vec![(0, rq.clone())]; // id 0 = the query itself
    let mut rs = 42u64;
    for id in 1..N {
        random.push((id, vector(&mut rs)));
    }
    let random_anchors: Vec<Vec<f32>> = {
        let mut a = 7u64;
        (0..ANCHORS).map(|_| vector(&mut a)).collect()
    };
    let recall_random = recall(&random, &rq, random_anchors, &exact_top(&random, &rq));

    // --- Realistic: topically-clustered memories added in batches (as an agent would:
    //     a run of trip notes, then a run of code notes, ...). Cluster c owns a contiguous
    //     id block, so the FIRST-N-as-anchors placeholder samples only the earliest clusters. ---
    let centres: Vec<Vec<f32>> = {
        let mut c = 99u64;
        (0..CLUSTERS).map(|_| vector(&mut c)).collect()
    };
    let mut s = 5u64;
    let cq = near(&centres[0], &mut s); // query lives in cluster 0
    let mut clustered: Vec<(u64, Vec<f32>)> = vec![(0, cq.clone())]; // id 0 = the query itself
    for id in 1..N {
        let c = id as usize * CLUSTERS / N as usize; // contiguous blocks → topical insertion order
        clustered.push((id, near(&centres[c], &mut s)));
    }
    let oracle = exact_top(&clustered, &cq);
    let top = *oracle.first().unwrap();

    let corpus_vecs: Vec<Vec<f32>> = clustered.iter().map(|(_, v)| v.clone()).collect();
    let first_n: Vec<Vec<f32>> = clustered
        .iter()
        .take(ANCHORS)
        .map(|(_, v)| v.clone())
        .collect();
    let kmeans = kmeans_anchors(&corpus_vecs, ANCHORS, KMEANS_ITERS);
    let recall_first_n = recall(&clustered, &cq, first_n, &oracle);
    let recall_kmeans = recall(&clustered, &cq, kmeans.clone(), &oracle);

    // Timing on the realistic (clustered + k-means) configuration.
    let mut exact = VectorIndex::new(DIMS);
    let mut ivf = IvfIndex::new(DIMS, kmeans);
    for (id, v) in &clustered {
        exact.insert(*id, v.clone()).unwrap();
        ivf.insert(*id, v.clone()).unwrap();
    }
    let exact_ns_per_op = time_search(|| {
        exact.search(&cq, K);
    }) as f64
        / N as f64;
    let ivf_ns_per_op = time_search(|| {
        ivf.search(&cq, K, PROBE);
    }) as f64
        / N as f64;

    println!(
        "{{\"benchmark\":\"mnema\",\"dims\":{DIMS},\"n\":{N},\"k\":{K},\"anchors\":{ANCHORS},\"probe\":{PROBE},\"clusters\":{CLUSTERS},\"top\":{top},\
         \"recall_random\":{recall_random:.3},\"recall_clustered_firstN\":{recall_first_n:.3},\"recall_clustered_kmeans\":{recall_kmeans:.3},\
         \"exact_ns_per_op\":{exact_ns_per_op:.3},\"ivf_ns_per_op\":{ivf_ns_per_op:.3}}}"
    );
}
