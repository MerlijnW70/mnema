//! Vector retrieval — Phase-2 slice A (`docs/proposals/mnema-memory-layer.md` §3.3).
//! The pluggable `Embedder` seam promised by ADR-0020 (bring-your-own model, so the
//! heavy ML dependency is the caller's choice, never ours) plus an **exact** cosine
//! `VectorIndex`.
//!
//! Exact, not approximate, on purpose. An HNSW / ANN index is *faster* but only
//! *approximately* correct — its win is latency, which is invisible to the mutation gate's
//! behavioural ratchet (BND-performance-blindness) and belongs on a channel-B fitness
//! benchmark, not here. Exact brute-force nearest-neighbour is the *correctness*
//! baseline this slice pins: given a query, it returns the true top-k by cosine
//! similarity, every time. ANN can later be added as a measured speed optimisation
//! that must match this oracle within a recall target — it does not replace it.
//!
//! Pure safe Rust, zero dependencies (ADR-0007 holds; no `secure` feature).

use std::cmp::Ordering;

use crate::MemoryId;

/// A text-to-vector embedder. Implemented by the caller (ADR-0020's pluggable seam):
/// Mnema never bundles an embedding model, so a local gguf/ONNX model — or a stub —
/// is the caller's choice. All vectors handed to one [`VectorIndex`] must share `dims`.
pub trait Embedder {
    /// Embed `text` into a fixed-length vector of `dims()` elements.
    ///
    /// # Degradation contract
    ///
    /// `embed` is infallible by signature, so an implementation that cannot produce a real
    /// embedding — a model that failed to run, an endpoint that refused the connection, a
    /// response of the wrong width — **must return `vec![0.0; dims()]`** rather than panic or
    /// return a short vector. Two guarantees rest on that:
    ///
    /// * the index keeps a consistent width, so one bad call cannot corrupt it; and
    /// * [`cosine`] scores a zero-magnitude vector at exactly `0.0`, never `NaN`, so a degraded
    ///   embedding cannot poison ranking.
    ///
    /// The memory is still **stored and still reachable** — hybrid retrieval finds it through
    /// the lexical channel; only the semantic channel is blind to it until it is re-embedded.
    /// Because that loss is invisible in the returned value, an implementation that degrades
    /// **must say so on stderr, including the cause**: a silent zero vector is indistinguishable
    /// from a genuine one, and the operator is the only party who can fix the endpoint or the
    /// model.
    fn embed(&self, text: &str) -> Vec<f32>;
    /// The dimensionality every embedding from this embedder has.
    fn dims(&self) -> usize;
}

/// What can go wrong feeding an index.
#[derive(Debug, PartialEq, Eq)]
pub enum VectorError {
    /// A vector whose length does not match the index's fixed dimensionality.
    DimMismatch,
    /// An [`IvfIndex`] with no anchors was asked to place a vector — it has no bucket.
    NoAnchors,
}

/// One scored search hit: a memory id and its cosine similarity to the query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scored {
    pub id: MemoryId,
    pub score: f32,
}

/// An exact nearest-neighbour index over fixed-dimension embeddings.
// No `Default`: a default (`dims == 0`) index rejects every real embedding with `DimMismatch`,
// so it is a public trap. Construct with [`VectorIndex::new`] against a real width instead.
#[derive(Clone, Debug)]
pub struct VectorIndex {
    dims: usize,
    /// `(id, vector, ‖vector‖)` — the norm is cached at insert so a query never recomputes it.
    entries: Vec<(MemoryId, Vec<f32>, f32)>,
}

impl VectorIndex {
    /// A new index over `dims`-dimensional vectors.
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            entries: Vec::new(),
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate the indexed `(id, vector)` pairs in insertion order. Lets a caller reuse the
    /// embeddings already computed here — e.g. to build the approximate [`IvfIndex`] without
    /// re-embedding, which with a real (model / HTTP) embedder would be a forward pass per vector.
    pub fn entries(&self) -> impl Iterator<Item = (MemoryId, &[f32])> {
        self.entries.iter().map(|(id, v, _)| (*id, v.as_slice()))
    }

    /// Index `vector` under `id`. Rejects a vector of the wrong dimensionality — a
    /// silent mismatch would let `zip` truncate and score against a truncated vector.
    pub fn insert(&mut self, id: MemoryId, vector: Vec<f32>) -> Result<(), VectorError> {
        if vector.len() != self.dims {
            return Err(VectorError::DimMismatch);
        }
        let vn = norm(&vector);
        self.entries.push((id, vector, vn));
        Ok(())
    }

    /// Remove every entry indexed under `id`, returning how many were dropped. Used
    /// when a memory is forgotten (hard-deleted) so a purged vector can never be
    /// surfaced by a later search.
    pub fn remove(&mut self, id: MemoryId) -> usize {
        let before = self.entries.len();
        self.entries.retain(|(entry_id, _, _)| *entry_id != id);
        before - self.entries.len()
    }

    /// The exact top-`k` hits for `query`, highest cosine similarity first. A query of
    /// the wrong dimensionality, or `k == 0`, yields no hits. Ties are broken by ascending
    /// id (a total order, so the approximate [`IvfIndex`] can match this exactly).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Scored> {
        if query.len() != self.dims {
            return Vec::new();
        }
        let qn = norm(query);
        // A zero-magnitude query has no direction: every cosine is 0.0, so "top-k" would be k
        // arbitrary ids whose tie-broken order then VOTES in retrieval fusion — a degraded
        // embedder (the documented zero-vector fallback) would outvote real lexical matches
        // with noise. No signal in, no hits out.
        if qn == 0.0 {
            return Vec::new();
        }
        let mut scored: Vec<Scored> = self
            .entries
            .iter()
            .map(|(id, v, vn)| Scored {
                id: *id,
                score: cosine_pre(query, qn, v, *vn),
            })
            .collect();
        // Descending by score, ties broken by ascending id — a *total* order independent
        // of iteration order, so the exact index and the IVF index (which collects
        // candidates in bucket order, not insertion order) agree even on tied scores. A
        // NaN (which cannot arise here — cosine guards it) sorts as Equal.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        scored.truncate(k);
        scored
    }
}

/// An **approximate** nearest-neighbour index (inverted file / IVF): vectors are
/// partitioned into buckets by their nearest anchor, and a query scans only the
/// `probe` buckets whose anchors are closest to it. This trades *recall* for *speed* —
/// the O(N) exact scan becomes O(probe · N/anchors) on average.
///
/// Recall is a **channel-B** property (statistical, benchmarked — cf. the Bloom
/// filter's FP-rate, Part 18/BND-statistical-quality), *not* a ratchet invariant. What
/// the ratchet *does* pin is the deterministic contract: with `probe >= anchors`, IVF
/// scans every bucket and returns **exactly** what the exact [`VectorIndex`] would —
/// ties included, since both rank by the same total order (score descending, then id
/// ascending) rather than a collection-order-dependent stable sort. A mutant that drops a
/// candidate bucket, mis-assigns a vector, or changes the tiebreak breaks that equality.
// No `Default`: like [`VectorIndex`], a `dims == 0` index is unusable; build with `IvfIndex::new`.
#[derive(Clone, Debug)]
pub struct IvfIndex {
    dims: usize,
    anchors: Vec<Vec<f32>>,
    /// Each bucket holds `(id, vector, ‖vector‖)` — the norm is cached at insert (as in
    /// [`VectorIndex`]) so a probed vector is scored without recomputing its magnitude.
    buckets: Vec<Vec<(MemoryId, Vec<f32>, f32)>>,
}

impl IvfIndex {
    /// A new index whose `anchors` (each `dims` long) define the buckets. At least one
    /// anchor is required for [`insert`](IvfIndex::insert) to place a vector.
    pub fn new(dims: usize, anchors: Vec<Vec<f32>>) -> Self {
        let buckets = vec![Vec::new(); anchors.len()];
        Self {
            dims,
            anchors,
            buckets,
        }
    }

    /// Number of indexed vectors, across all buckets.
    pub fn len(&self) -> usize {
        self.buckets.iter().map(Vec::len).sum()
    }

    /// Whether no vectors are indexed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index `vector` under `id`, in the bucket of its nearest anchor.
    pub fn insert(&mut self, id: MemoryId, vector: Vec<f32>) -> Result<(), VectorError> {
        if vector.len() != self.dims {
            return Err(VectorError::DimMismatch);
        }
        let bucket = self.nearest_anchor(&vector).ok_or(VectorError::NoAnchors)?;
        let vn = norm(&vector);
        self.buckets[bucket].push((id, vector, vn));
        Ok(())
    }

    /// The index of the anchor most similar to `vector`, or `None` if there are no
    /// anchors. Ties keep the earliest anchor.
    fn nearest_anchor(&self, vector: &[f32]) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, anchor) in self.anchors.iter().enumerate() {
            let sim = cosine(vector, anchor);
            // `is_none_or` avoids the `best.is_none() || sim > best.unwrap().1` unwrap: no fail-open site to justify.
            if best.is_none_or(|(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }
        best.map(|(i, _)| i)
    }

    /// The approximate top-`k` hits for `query`, scanning the `probe` buckets whose
    /// anchors are nearest the query. With `probe >= anchors.len()` this scans every
    /// bucket and equals the exact result. A wrong-dimension query, or `k == 0`, yields
    /// nothing.
    pub fn search(&self, query: &[f32], k: usize, probe: usize) -> Vec<Scored> {
        if query.len() != self.dims {
            return Vec::new();
        }
        // No direction, no hits — mirrors `VectorIndex::search` (see there): a zero-magnitude
        // query must not emit k arbitrary tie-broken ids as if they were dense matches.
        if norm(query) == 0.0 {
            return Vec::new();
        }
        // Rank anchors by similarity to the query; take the `probe` closest.
        let mut ranked: Vec<(usize, f32)> = self
            .anchors
            .iter()
            .enumerate()
            .map(|(i, a)| (i, cosine(query, a)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let qn = norm(query);
        let mut scored: Vec<Scored> = Vec::new();
        for (bucket, _) in ranked.iter().take(probe) {
            for (id, v, vn) in &self.buckets[*bucket] {
                scored.push(Scored {
                    id: *id,
                    score: cosine_pre(query, qn, v, *vn),
                });
            }
        }
        // Same total order as the exact index (score desc, then id asc), so with
        // `probe >= anchors` this returns byte-for-byte what `VectorIndex::search` would —
        // ties included.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        scored.truncate(k);
        scored
    }
}

/// Train up to `k` anchor centroids over `vectors` with **deterministic** k-means (spherical:
/// assignment is by cosine, matching how [`IvfIndex`] buckets a vector to its nearest anchor),
/// running `iters` Lloyd iterations. Trained anchors sit at the data's actual cluster centres, so
/// an [`IvfIndex`] built on them buckets neighbours together and recovers far more of the exact
/// top-k than anchors seeded from arbitrary points — the recall lever measured in the bench.
///
/// Determinism (no rng) is deliberate: the result is a pure function of the input, so the anchor
/// set is reproducible and every branch here stays mutation-provable. Initialisation samples the
/// corpus at even strides; an empty cluster keeps its centroid rather than dividing by zero.
/// `k` is clamped to the corpus size; an empty corpus or `k == 0` yields no anchors.
pub fn kmeans_anchors(vectors: &[Vec<f32>], k: usize, iters: usize) -> Vec<Vec<f32>> {
    let k = k.min(vectors.len());
    if k == 0 {
        return Vec::new();
    }
    let dims = vectors[0].len();
    // Even-stride initialisation: spread the initial centroids across the corpus deterministically.
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| vectors[i * vectors.len() / k].clone())
        .collect();

    for _ in 0..iters {
        let mut sums = vec![vec![0.0f32; dims]; k];
        let mut counts = vec![0usize; k];
        for v in vectors {
            let c = nearest_centroid(v, &centroids);
            for d in 0..dims {
                sums[c][d] += v[d];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            // An empty cluster keeps its previous centroid — dividing a zero sum by a zero count
            // would be NaN and poison every future assignment to it.
            if counts[c] > 0 {
                for d in 0..dims {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }
    centroids
}

/// The index of the centroid most similar (by cosine) to `v`; ties keep the earliest centroid.
fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_sim = f32::NEG_INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let sim = cosine(v, c);
        if sim > best_sim {
            best_sim = sim;
            best = i;
        }
    }
    best
}

/// L2 magnitude `√(Σ xᵢ²)`. Precomputed once per stored vector (and once per query) so that
/// scoring a candidate is a dot product plus a division, instead of re-deriving both
/// magnitudes on every comparison — the same arithmetic as [`cosine`], just not repeated.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity from precomputed norms: `dot / (na · nb)`, guarding a zero magnitude to
/// `0.0` exactly as [`cosine`] does. Bit-identical to `cosine(a, b)` when `na == norm(a)` and
/// `nb == norm(b)` — the norms are the only thing hoisted out of the inner loop.
fn cosine_pre(a: &[f32], na: f32, b: &[f32], nb: f32) -> f32 {
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        dot / (na * nb)
    }
}

/// Cosine similarity of two equal-length vectors, in `[-1, 1]`. Returns `0.0` when
/// either vector has zero magnitude (the angle is undefined) rather than `NaN`. Used where a
/// norm is not worth caching (anchor ranking, k-means); the query hot path uses [`cosine_pre`].
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_yields_indexed_vectors_in_insertion_order() {
        let mut idx = VectorIndex::new(2);
        idx.insert(7, vec![1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.5, 0.5]).unwrap();
        let got: Vec<(MemoryId, Vec<f32>)> =
            idx.entries().map(|(id, v)| (id, v.to_vec())).collect();
        assert_eq!(got, vec![(7, vec![1.0, 0.0]), (3, vec![0.5, 0.5])]);
        // A removed vector is no longer yielded.
        idx.remove(7);
        let ids: Vec<MemoryId> = idx.entries().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![3]);
    }

    /// A tiny deterministic embedder that proves the pluggable seam is real: it maps
    /// text to `[vowel_count, consonant_count]`. Not meaningful — just exercisable.
    struct VowelEmbedder;
    impl Embedder for VowelEmbedder {
        fn dims(&self) -> usize {
            2
        }
        fn embed(&self, text: &str) -> Vec<f32> {
            let vowels = text.chars().filter(|c| "aeiou".contains(*c)).count() as f32;
            let letters = text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f32;
            vec![vowels, letters - vowels]
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn cosine_is_one_for_parallel_and_zero_for_orthogonal() {
        assert!(approx(cosine(&[1.0, 0.0], &[1.0, 0.0]), 1.0));
        assert!(approx(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0));
        assert!(approx(cosine(&[2.0, 0.0], &[5.0, 0.0]), 1.0)); // scale-invariant
        assert!(approx(cosine(&[1.0, 0.0], &[-1.0, 0.0]), -1.0)); // anti-parallel
    }

    #[test]
    fn a_zero_magnitude_vector_scores_zero_not_nan() {
        // Pins the `||` guard: with `&&`, one zero vector would divide by zero → NaN.
        let s = cosine(&[0.0, 0.0], &[1.0, 1.0]);
        assert!(approx(s, 0.0));
        assert!(!s.is_nan());
    }

    #[test]
    fn a_zero_magnitude_query_returns_no_hits_from_either_index() {
        // A degraded embedder's documented fallback is the zero vector. It has no direction, so
        // every cosine ties at 0.0 — "top-k" would be k arbitrary ids that then vote as dense
        // matches in retrieval fusion, outranking real lexical hits with pure noise. No signal
        // in, no hits out — from the exact index and the IVF index alike.
        let mut idx = VectorIndex::new(2);
        idx.insert(1, vec![1.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0]).unwrap();
        assert!(idx.search(&[0.0, 0.0], 2).is_empty());

        let mut ivf = IvfIndex::new(2, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        ivf.insert(1, vec![1.0, 0.0]).unwrap();
        ivf.insert(2, vec![0.0, 1.0]).unwrap();
        assert!(ivf.search(&[0.0, 0.0], 2, 2).is_empty());
    }

    #[test]
    fn search_ranks_by_similarity_descending() {
        let mut idx = VectorIndex::new(2);
        idx.insert(1, vec![1.0, 0.0]).unwrap(); // parallel to query → best
        idx.insert(2, vec![0.0, 1.0]).unwrap(); // orthogonal → worst
        idx.insert(3, vec![1.0, 1.0]).unwrap(); // 45° → middle
        let hits = idx.search(&[1.0, 0.0], 3);
        let ids: Vec<MemoryId> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![1, 3, 2]);
        assert!(approx(hits[0].score, 1.0));
        assert!(approx(hits[2].score, 0.0));
    }

    #[test]
    fn search_returns_at_most_k_hits() {
        let mut idx = VectorIndex::new(2);
        idx.insert(1, vec![1.0, 0.0]).unwrap();
        idx.insert(2, vec![1.0, 1.0]).unwrap();
        idx.insert(3, vec![0.0, 1.0]).unwrap();
        assert_eq!(idx.search(&[1.0, 0.0], 2).len(), 2);
        assert_eq!(
            idx.search(&[1.0, 0.0], 2)
                .iter()
                .map(|h| h.id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(idx.search(&[1.0, 0.0], 0).is_empty());
        assert_eq!(idx.search(&[1.0, 0.0], 99).len(), 3); // k past the end is clamped
    }

    #[test]
    fn a_wrong_dimension_vector_is_rejected_on_insert() {
        let mut idx = VectorIndex::new(2);
        assert_eq!(idx.insert(1, vec![1.0]), Err(VectorError::DimMismatch));
        assert_eq!(
            idx.insert(2, vec![1.0, 2.0, 3.0]),
            Err(VectorError::DimMismatch)
        );
        assert!(idx.insert(3, vec![1.0, 2.0]).is_ok());
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());
    }

    #[test]
    fn remove_drops_only_the_matching_id() {
        let mut idx = VectorIndex::new(2);
        idx.insert(1, vec![1.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0]).unwrap();
        idx.insert(3, vec![1.0, 1.0]).unwrap();
        assert_eq!(idx.remove(2), 1); // one dropped
        assert_eq!(idx.len(), 2);
        let ids: Vec<MemoryId> = idx.search(&[0.0, 1.0], 9).iter().map(|h| h.id).collect();
        assert!(!ids.contains(&2)); // and it can no longer be found
        assert!(ids.contains(&1) && ids.contains(&3)); // the others survive
        assert_eq!(idx.remove(99), 0); // an absent id drops nothing
    }

    #[test]
    fn a_wrong_dimension_query_yields_no_hits() {
        let mut idx = VectorIndex::new(2);
        idx.insert(1, vec![1.0, 0.0]).unwrap();
        assert!(idx.search(&[1.0], 5).is_empty());
        assert!(idx.search(&[1.0, 2.0, 3.0], 5).is_empty());
    }

    #[test]
    fn end_to_end_through_a_pluggable_embedder() {
        let embed = VowelEmbedder;
        let mut idx = VectorIndex::new(embed.dims());
        idx.insert(1, embed.embed("aeiou")).unwrap(); // [5, 0]
        idx.insert(2, embed.embed("xyz")).unwrap(); // [0, 3]
        // A vowel-heavy query should rank the vowel-heavy memory first.
        let hits = idx.search(&embed.embed("aiu"), 2); // [3, 0]
        assert_eq!(hits[0].id, 1);
    }

    // ---- IvfIndex (approximate) ----

    fn ivf_axis() -> IvfIndex {
        // Two buckets: the x-axis and the y-axis.
        let mut idx = IvfIndex::new(2, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        idx.insert(1, vec![1.0, 0.1]).unwrap(); // → bucket 0
        idx.insert(2, vec![1.0, 0.2]).unwrap(); // → bucket 0
        idx.insert(3, vec![0.1, 1.0]).unwrap(); // → bucket 1
        idx.insert(4, vec![0.2, 1.0]).unwrap(); // → bucket 1
        idx
    }

    #[test]
    fn ivf_assigns_a_vector_to_its_nearest_anchor() {
        let idx = ivf_axis();
        // Pins the argmax in `nearest_anchor` (a `>`→`<` flip picks the far anchor).
        assert_eq!(idx.nearest_anchor(&[0.9, 0.1]), Some(0));
        assert_eq!(idx.nearest_anchor(&[0.1, 0.9]), Some(1));
        // A vector equidistant from both anchors keeps the EARLIER one — pins `>` vs
        // `>=` (the latter would switch to the later anchor on a tie).
        assert_eq!(idx.nearest_anchor(&[1.0, 1.0]), Some(0));
        assert_eq!(idx.len(), 4);
    }

    #[test]
    fn ivf_full_probe_equals_the_exact_oracle() {
        let ivf = ivf_axis();
        let mut exact = VectorIndex::new(2);
        exact.insert(1, vec![1.0, 0.1]).unwrap();
        exact.insert(2, vec![1.0, 0.2]).unwrap();
        exact.insert(3, vec![0.1, 1.0]).unwrap();
        exact.insert(4, vec![0.2, 1.0]).unwrap();

        let query = [1.0, 0.0];
        let exact_ids: Vec<MemoryId> = exact.search(&query, 4).iter().map(|h| h.id).collect();
        // probe == anchors → every bucket scanned → must equal the exact ranking.
        let ivf_ids: Vec<MemoryId> = ivf.search(&query, 4, 2).iter().map(|h| h.id).collect();
        assert_eq!(ivf_ids, exact_ids);
        assert_eq!(exact_ids, vec![1, 2, 4, 3]);
    }

    #[test]
    fn exact_search_breaks_score_ties_by_ascending_id() {
        // Two identical vectors inserted in DESCENDING id order tie on cosine. The result
        // must order them by id (5 before 6), not by insertion order (which would give 6, 5)
        // — this is the total order that lets the IVF match the exact index on ties.
        let mut idx = VectorIndex::new(2);
        idx.insert(6, vec![1.0, 0.0]).unwrap();
        idx.insert(5, vec![1.0, 0.0]).unwrap();
        let ids: Vec<MemoryId> = idx.search(&[1.0, 0.0], 2).iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![5, 6]);
    }

    #[test]
    fn ivf_search_breaks_score_ties_by_ascending_id_like_the_exact_index() {
        // Same tie, in the IVF: both land in bucket 0, collected in insertion order (6, 5);
        // the id tiebreak must reorder them to (5, 6) so IVF and exact never diverge on ties.
        let mut ivf = IvfIndex::new(2, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        ivf.insert(6, vec![1.0, 0.0]).unwrap();
        ivf.insert(5, vec![1.0, 0.0]).unwrap();
        let ids: Vec<MemoryId> = ivf.search(&[1.0, 0.0], 2, 2).iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![5, 6]);
    }

    #[test]
    fn ivf_partial_probe_scans_only_the_nearest_bucket() {
        let ivf = ivf_axis();
        // probe == 1 → only the x-axis bucket; the top hit is its best member, and no
        // y-axis id appears. (This is the recall/speed trade — a channel-B property.)
        let hits = ivf.search(&[1.0, 0.0], 4, 1);
        let ids: Vec<MemoryId> = hits.iter().map(|h| h.id).collect();
        assert_eq!(hits[0].id, 1);
        assert!(ids.iter().all(|id| *id == 1 || *id == 2));
    }

    #[test]
    fn ivf_rejects_a_wrong_dimension_vector() {
        let mut idx = IvfIndex::new(2, vec![vec![1.0, 0.0]]);
        assert_eq!(
            idx.insert(1, vec![1.0, 2.0, 3.0]),
            Err(VectorError::DimMismatch)
        );
    }

    #[test]
    fn ivf_with_no_anchors_cannot_place_a_vector() {
        let mut idx = IvfIndex::new(2, vec![]);
        assert_eq!(idx.insert(1, vec![1.0, 0.0]), Err(VectorError::NoAnchors));
        assert!(idx.is_empty());
    }

    #[test]
    fn ivf_search_with_a_wrong_dimension_query_yields_nothing() {
        // Pins the dim guard in IVF `search`: dropping it would `zip`-truncate and
        // score against a mismatched query rather than returning empty.
        let ivf = ivf_axis();
        assert!(ivf.search(&[1.0], 4, 2).is_empty());
    }

    #[test]
    fn nearest_centroid_picks_the_cosine_argmax_and_keeps_the_earlier_on_a_tie() {
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(nearest_centroid(&[0.9, 0.1], &centroids), 0);
        assert_eq!(nearest_centroid(&[0.1, 0.9], &centroids), 1);
        // Equidistant → the earlier centroid wins (pins `>` vs `>=`).
        assert_eq!(nearest_centroid(&[1.0, 1.0], &centroids), 0);
    }

    #[test]
    fn kmeans_recovers_the_cluster_centres() {
        // Two obvious clusters near the x- and y-axes. Even-stride init seeds centroids at
        // index 0 ([1,0]) and 2 ([0,1]); the means converge to the per-cluster averages.
        let data = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let cs = kmeans_anchors(&data, 2, 5);
        assert_eq!(cs.len(), 2);
        // Cluster 0 ≈ mean([1,0],[0.9,0.1]) = [0.95, 0.05]; cluster 1 ≈ [0.05, 0.95]. Exact
        // values pin the mean division (a missing `/count` would leave the raw sums).
        assert!(
            (cs[0][0] - 0.95).abs() < 1e-4 && (cs[0][1] - 0.05).abs() < 1e-4,
            "{cs:?}"
        );
        assert!(
            (cs[1][0] - 0.05).abs() < 1e-4 && (cs[1][1] - 0.95).abs() < 1e-4,
            "{cs:?}"
        );
    }

    #[test]
    fn kmeans_leaves_an_empty_clusters_centroid_unchanged_not_nan() {
        // Identical vectors all fall in the first centroid; the second gets nothing. Its
        // centroid must stay at its init value, not become NaN by dividing a zero sum by 0.
        let data = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let cs = kmeans_anchors(&data, 2, 3);
        assert_eq!(cs.len(), 2);
        assert_eq!(
            cs[1],
            vec![1.0, 0.0],
            "empty cluster kept its centroid (no NaN)"
        );
    }

    #[test]
    fn cosine_pre_equals_cosine_and_guards_a_zero_norm_on_either_side() {
        let a = [1.0, 2.0, 3.0];
        let b = [3.0, 2.0, 1.0];
        let zero = [0.0, 0.0, 0.0];
        // With true norms it is bit-identical to `cosine` (pins that `norm` is √Σx², and that
        // the hot path did not change any score).
        assert_eq!(cosine_pre(&a, norm(&a), &b, norm(&b)), cosine(&a, &b));
        // A zero magnitude on EITHER side yields 0.0, never NaN — pins the guard and the `||`
        // (a `&&` or a dropped guard would divide by zero here).
        assert_eq!(cosine_pre(&a, norm(&a), &zero, norm(&zero)), 0.0);
        assert_eq!(cosine_pre(&zero, norm(&zero), &a, norm(&a)), 0.0);
    }

    #[test]
    fn kmeans_clamps_k_and_handles_degenerate_input() {
        assert!(kmeans_anchors(&[], 3, 5).is_empty()); // empty corpus
        assert!(kmeans_anchors(&[vec![1.0, 0.0]], 0, 5).is_empty()); // k == 0
        // k larger than the corpus is clamped to the corpus size.
        assert_eq!(
            kmeans_anchors(&[vec![1.0, 0.0], vec![0.0, 1.0]], 5, 3).len(),
            2
        );
    }
}
