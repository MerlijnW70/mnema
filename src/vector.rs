//! Vector retrieval — Phase-2 slice A (`docs/proposals/engram-memory-layer.md` §3.3).
//! The pluggable [`Embedder`] seam promised by ADR-0020 (bring-your-own model, so the
//! heavy ML dependency is the caller's choice, never ours) plus an **exact** cosine
//! [`VectorIndex`].
//!
//! Exact, not approximate, on purpose. An HNSW / ANN index is *faster* but only
//! *approximately* correct — its win is latency, which is invisible to internal-tool's
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
/// Engram never bundles an embedding model, so a local gguf/ONNX model — or a stub —
/// is the caller's choice. All vectors handed to one [`VectorIndex`] must share `dims`.
pub trait Embedder {
    /// Embed `text` into a fixed-length vector of `dims()` elements.
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
#[derive(Clone, Debug, Default)]
pub struct VectorIndex {
    dims: usize,
    entries: Vec<(MemoryId, Vec<f32>)>,
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

    /// Index `vector` under `id`. Rejects a vector of the wrong dimensionality — a
    /// silent mismatch would let `zip` truncate and score against a truncated vector.
    pub fn insert(&mut self, id: MemoryId, vector: Vec<f32>) -> Result<(), VectorError> {
        if vector.len() != self.dims {
            return Err(VectorError::DimMismatch);
        }
        self.entries.push((id, vector));
        Ok(())
    }

    /// Remove every entry indexed under `id`, returning how many were dropped. Used
    /// when a memory is forgotten (hard-deleted) so a purged vector can never be
    /// surfaced by a later search.
    pub fn remove(&mut self, id: MemoryId) -> usize {
        let before = self.entries.len();
        self.entries.retain(|(entry_id, _)| *entry_id != id);
        before - self.entries.len()
    }

    /// The exact top-`k` hits for `query`, highest cosine similarity first. A query of
    /// the wrong dimensionality, or `k == 0`, yields no hits. Ties are broken by ascending
    /// id (a total order, so the approximate [`IvfIndex`] can match this exactly).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Scored> {
        if query.len() != self.dims {
            return Vec::new();
        }
        let mut scored: Vec<Scored> = self
            .entries
            .iter()
            .map(|(id, v)| Scored {
                id: *id,
                score: cosine(query, v),
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
#[derive(Clone, Debug, Default)]
pub struct IvfIndex {
    dims: usize,
    anchors: Vec<Vec<f32>>,
    buckets: Vec<Vec<(MemoryId, Vec<f32>)>>,
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
        self.buckets[bucket].push((id, vector));
        Ok(())
    }

    /// The index of the anchor most similar to `vector`, or `None` if there are no
    /// anchors. Ties keep the earliest anchor.
    fn nearest_anchor(&self, vector: &[f32]) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, anchor) in self.anchors.iter().enumerate() {
            let sim = cosine(vector, anchor);
            if best.is_none() || sim > best.unwrap().1 {
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
        // Rank anchors by similarity to the query; take the `probe` closest.
        let mut ranked: Vec<(usize, f32)> = self
            .anchors
            .iter()
            .enumerate()
            .map(|(i, a)| (i, cosine(query, a)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut scored: Vec<Scored> = Vec::new();
        for (bucket, _) in ranked.iter().take(probe) {
            for (id, v) in &self.buckets[*bucket] {
                scored.push(Scored {
                    id: *id,
                    score: cosine(query, v),
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

/// Cosine similarity of two equal-length vectors, in `[-1, 1]`. Returns `0.0` when
/// either vector has zero magnitude (the angle is undefined) rather than `NaN`.
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
}
