//! Scale sweep — how exact vs approximate (IVF) recall behaves as the corpus grows.
//!
//! The `engram` bench answers "exact vs IVF at one size"; this answers "how does the gap
//! move with N". For each N it reports the *raw per-search latency* (microseconds, not
//! per-op) of the exact O(N) scan and the k-means-anchored IVF, plus the speedup and the
//! IVF's recall of the exact top-k. The workload is the realistic clustered one (topically
//! batched), so recall reflects real embeddings rather than structureless noise.
//!
//! Deliberately not wired into the fitness signal (`scripts/fitness-engram.sh`) — building a
//! 100k-vector index + training anchors is seconds of work. Run on demand:
//! `cargo bench --bench scale`.

use engram::vector::{IvfIndex, VectorIndex, kmeans_anchors};
use std::time::Instant;

const DIMS: usize = 64;
const K: usize = 10;
const SPREAD: f32 = 0.10;
const KMEANS_ITERS: usize = 5;
const SIZES: [u64; 5] = [1_000, 5_000, 20_000, 50_000, 100_000];

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

fn near(centre: &[f32], state: &mut u64) -> Vec<f32> {
    centre
        .iter()
        .map(|&x| x + SPREAD * (splitmix(state) - 0.5))
        .collect()
}

/// Median raw latency of one `search` call, in microseconds.
fn micros(mut run: impl FnMut()) -> f64 {
    for _ in 0..3 {
        run();
    }
    let mut samples = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        run();
        samples.push(t.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1000.0
}

/// Integer square root — anchor count ≈ √N, the usual IVF bucket heuristic.
fn isqrt(n: u64) -> usize {
    (n as f64).sqrt() as usize
}

fn main() {
    println!("        N   anchors  probe   exact_us    ivf_us   speedup   recall");
    for &n in &SIZES {
        let anchors_k = isqrt(n).max(1);
        let probe = (anchors_k / 8).max(1);

        // Realistic clustered corpus: cluster count ≈ anchor count, topically batched.
        let clusters = anchors_k;
        let centres: Vec<Vec<f32>> = {
            let mut c = 99u64;
            (0..clusters).map(|_| vector(&mut c)).collect()
        };
        let mut s = 5u64;
        let query = near(&centres[0], &mut s);
        let mut corpus: Vec<(u64, Vec<f32>)> = vec![(0, query.clone())];
        for id in 1..n {
            let c = id as usize * clusters / n as usize;
            corpus.push((id, near(&centres[c], &mut s)));
        }

        let mut exact = VectorIndex::new(DIMS);
        for (id, v) in &corpus {
            exact.insert(*id, v.clone()).unwrap();
        }
        let oracle: Vec<u64> = exact.search(&query, K).iter().map(|h| h.id).collect();

        let corpus_vecs: Vec<Vec<f32>> = corpus.iter().map(|(_, v)| v.clone()).collect();
        let anchors = kmeans_anchors(&corpus_vecs, anchors_k, KMEANS_ITERS);
        let mut ivf = IvfIndex::new(DIMS, anchors);
        for (id, v) in &corpus {
            ivf.insert(*id, v.clone()).unwrap();
        }

        let got: Vec<u64> = ivf.search(&query, K, probe).iter().map(|h| h.id).collect();
        let recall = oracle.iter().filter(|id| got.contains(id)).count() as f64 / K as f64;

        let exact_us = micros(|| {
            exact.search(&query, K);
        });
        let ivf_us = micros(|| {
            ivf.search(&query, K, probe);
        });

        println!(
            "{n:>9}   {anchors_k:>5}   {probe:>4}   {exact_us:>8.2}  {ivf_us:>8.2}   {:>6.1}x   {recall:>6.3}",
            exact_us / ivf_us
        );
    }
}
