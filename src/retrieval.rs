//! Hybrid retrieval — Phase-2 slice B (`docs/proposals/mnema-memory-layer.md` §3.3).
//! The read path that ties the pieces together: run several independent retrievers
//! (dense vector, recency, lexical keyword), **fuse** their rankings with reciprocal-
//! rank fusion, resolve the winners back to memories, and pack them through the
//! egress filter (ADR-0021) under a character budget.
//!
//! Why RRF and not a weighted score sum? Because the retrievers' raw scores are not
//! commensurable — a cosine in `[-1,1]` and a keyword overlap count live on different
//! scales, and normalising them invites a tuning knob that overfits. RRF fuses on
//! *rank position* alone, so it needs no per-retriever weight and rewards documents
//! that several retrievers agree on. Hybrid beating pure-vector is the whole point.
//!
//! Pure safe Rust, zero dependencies (ADR-0007 holds).

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use crate::vector::{Embedder, VectorIndex};
use crate::{BundleItem, Destination, Memory, MemoryId};

/// The reciprocal-rank-fusion constant (Cormack et al., 2009 use 60). It damps the
/// gap between top ranks so no single retriever's #1 can dominate the fusion.
pub const RRF_K: f32 = 60.0;

/// A forgetting curve for recall: a memory's fused score is scaled by its `importance`
/// times a recency weight that halves every `half_life` ticks of age (proposal §3.2).
/// Pass to [`hybrid_recall`] to prefer recent, important memories over stale ones.
#[derive(Clone, Copy, Debug)]
pub struct Decay {
    /// The current logical time; a memory's age is `now - at`.
    pub now: u64,
    /// Ticks of age at which the recency weight halves. `0` disables decay.
    pub half_life: u64,
}

/// The recency weight of a memory `age` ticks old under a `half_life`: `0.5^(age /
/// half_life)`, in `(0, 1]`. A `half_life` of `0` disables decay (weight `1.0`),
/// which also avoids a division by zero.
pub fn decay_weight(age: u64, half_life: u64) -> f32 {
    if half_life == 0 {
        return 1.0;
    }
    let elapsed_half_lives = age as f32 / half_life as f32;
    0.5_f32.powf(elapsed_half_lives)
}

/// The forgetting-curve score of a hit: its fused `base` score scaled *multiplicatively*
/// by the memory's `importance` and its recency weight. Both factors multiply — a salient
/// memory (`importance > 1`) and a fresh one (`weight → 1`) reinforce; a stale, dull one
/// sinks. Factored out (and asserted on exact values) so the *product* contract is pinned,
/// not merely the resulting order — an additive combination would rank differently.
pub fn decayed_score(base: f32, importance: f32, age: u64, half_life: u64) -> f32 {
    base * importance * decay_weight(age, half_life)
}

/// A fused hit: a memory id and its summed reciprocal-rank score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fused {
    pub id: MemoryId,
    pub score: f32,
}

/// Fuse several ranked id-lists into one ranking by reciprocal-rank fusion: each id
/// scores the sum over lists of `1 / (RRF_K + rank)`, with `rank` 1-based. Ids are
/// de-duplicated (their contributions summed) and returned highest-score first.
pub fn rrf_fuse(rankings: &[Vec<MemoryId>]) -> Vec<Fused> {
    let weighted: Vec<(f32, &[MemoryId])> = rankings.iter().map(|r| (1.0, r.as_slice())).collect();
    rrf_fuse_weighted(&weighted)
}

/// Per-retriever weights for hybrid fusion — how much each retriever's opinion counts. Each
/// retriever's reciprocal-rank contribution is scaled by its weight, so a caller can make the
/// dense (embedding) retriever outvote the lexical and recency ones. [`Default`] is all-`1.0`
/// (the balanced fusion `rrf_fuse` gives); [`semantic`](RetrievalWeights::semantic) tips it
/// toward meaning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetrievalWeights {
    /// Weight of the dense/embedding retriever (semantic similarity).
    pub dense: f32,
    /// Weight of the recency retriever (newest first).
    pub recency: f32,
    /// Weight of the keyword retriever (lexical overlap).
    pub keyword: f32,
}

impl Default for RetrievalWeights {
    fn default() -> Self {
        Self {
            dense: 1.0,
            recency: 1.0,
            keyword: 1.0,
        }
    }
}

impl RetrievalWeights {
    /// Favor the dense (embedding) retriever over lexical + recency, so a real semantic
    /// embedder's meaning-match wins over a memory that merely shares a word or is newer. Only
    /// worthwhile with a semantic embedder — with the lexical `HashEmbedder` the dense signal is
    /// mostly noise, so prefer [`Default`](RetrievalWeights::default) there.
    #[must_use]
    pub fn semantic() -> Self {
        Self {
            dense: 4.0,
            recency: 1.0,
            keyword: 1.0,
        }
    }
}

/// Weighted reciprocal-rank fusion: like [`rrf_fuse`], but each ranking carries a weight that
/// scales its `1 / (RRF_K + rank)` contribution. A memory's fused score is the weighted sum of
/// its contributions across the lists it appears in. `rrf_fuse` is the all-`1.0` special case.
pub fn rrf_fuse_weighted(rankings: &[(f32, &[MemoryId])]) -> Vec<Fused> {
    let mut acc: Vec<Fused> = Vec::new();
    for (weight, list) in rankings {
        for (pos, id) in list.iter().enumerate() {
            let rank = pos as f32 + 1.0;
            let contribution = weight / (RRF_K + rank);
            match acc.iter_mut().find(|f| f.id == *id) {
                Some(f) => f.score += contribution,
                None => acc.push(Fused {
                    id: *id,
                    score: contribution,
                }),
            }
        }
    }
    acc.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    acc
}

/// Okapi BM25 tuning constants (the standard defaults): `k1` controls term-frequency
/// saturation, `b` the document-length normalization.
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// The BM25 contribution of one query term to one document. `tf` = the term's frequency in
/// the document, `df` = number of documents containing it, `n` = corpus size, `dl` = this
/// document's length, `avgdl` = mean document length. Uses the Lucene BM25 IDF
/// `ln(1 + (n - df + 0.5)/(df + 0.5))`, which is always non-negative.
fn bm25_term_score(tf: f32, df: f32, n: f32, dl: f32, avgdl: f32) -> f32 {
    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
    let norm = tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl));
    idf * norm
}

/// Mean document length over the corpus, floored at `1.0` so BM25's length term stays finite
/// on a degenerate all-empty corpus (where, having no term matches, it is never consulted).
fn avg_len(total_len: usize, n: f32) -> f32 {
    (total_len as f32 / n).max(1.0)
}

/// Rank memories by **BM25** relevance to `query` — a rare query term outweighs a common one
/// (IDF), extra repetitions of a term give diminishing returns (`k1`), and a hit in a short
/// memory outweighs the same hit in a long one (`b`). Only memories containing at least one
/// query term are returned; ties keep input order (the sort is stable).
pub fn bm25_rank(query: &str, memories: &[Memory]) -> Vec<MemoryId> {
    let mut q_terms = tokenize(query);
    q_terms.sort();
    q_terms.dedup();

    // Tokenize each memory once, then gather the corpus statistics BM25 needs. Empty inputs —
    // no query terms, or no memories — simply produce no scored documents below (a term never
    // contributes, or there is nothing to score), so no special-case guard is needed.
    let docs: Vec<(MemoryId, Vec<String>)> = memories
        .iter()
        .map(|m| (m.id, tokenize(&m.content)))
        .collect();
    let n = docs.len() as f32;
    let total_len: usize = docs.iter().map(|(_, d)| d.len()).sum();
    let avgdl = avg_len(total_len, n);
    let df: Vec<f32> = q_terms
        .iter()
        .map(|t| docs.iter().filter(|(_, d)| d.contains(t)).count() as f32)
        .collect();

    let mut scored: Vec<(MemoryId, f32)> = docs
        .iter()
        .filter_map(|(id, doc)| {
            let dl = doc.len() as f32;
            let mut score = 0.0f32;
            for (t, &df_t) in q_terms.iter().zip(&df) {
                // A term absent from this document has tf = 0, so its BM25 contribution is 0;
                // summing it unconditionally is exact and needs no `tf > 0` guard.
                let tf = doc.iter().filter(|w| w.as_str() == t.as_str()).count() as f32;
                score += bm25_term_score(tf, df_t, n, dl, avgdl);
            }
            (score > 0.0).then_some((*id, score))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}

/// Lowercase, split on non-alphanumeric, drop empties. The query side of [`bm25_rank`]
/// de-duplicates these so a repeated query term contributes once; the document side keeps
/// repetitions, since BM25 scores term frequency.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// A memory at least this token-Jaccard-similar to an already-selected one is treated as a
/// near-duplicate and dropped from recall. Conservative — only near-identical memories are
/// suppressed, so distinct-but-related memories still surface.
const DEDUP_THRESHOLD: f32 = 0.8;

/// Token-set Jaccard similarity of two texts, in `[0, 1]`: `|shared tokens| / |all tokens|`.
/// Two empty texts are defined as identical (`1.0`).
fn content_similarity(a: &str, b: &str) -> f32 {
    let ta: BTreeSet<String> = tokenize(a).into_iter().collect();
    let tb: BTreeSet<String> = tokenize(b).into_iter().collect();
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    let intersection = ta.intersection(&tb).count() as f32;
    let union = ta.len() as f32 + tb.len() as f32 - intersection;
    intersection / union
}

/// Suppress near-duplicate memories from a relevance-ordered list: keep a memory only if it is
/// less than `threshold` similar (token Jaccard) to every memory already kept, so the recall
/// budget is not spent on repeats. Order is preserved — the first (most relevant) of a
/// near-duplicate pair wins.
fn dedup_similar<'a>(ordered: &[&'a Memory], threshold: f32) -> Vec<&'a Memory> {
    let mut kept: Vec<&Memory> = Vec::new();
    for &m in ordered {
        let is_dup = kept
            .iter()
            .any(|k| content_similarity(&m.content, &k.content) >= threshold);
        if !is_dup {
            kept.push(m);
        }
    }
    kept
}

/// Hybrid recall: embed the query, gather the top `per_retriever` ids from each of the
/// dense (vector), recency, and lexical (keyword) retrievers, fuse them with RRF, and
/// pack the fused order through the egress filter under `char_budget`.
///
/// `index` and `memories` must share an id space (the vector index is keyed by the
/// same [`MemoryId`] the memories carry). The returned bundle is egress-safe for
/// `dest` by construction — it goes through the same `pack_bundle` choke point as
/// recency assembly, so a `Private` memory never reaches a `Remote` bundle.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_recall(
    query: &str,
    memories: &[Memory],
    index: &VectorIndex,
    embedder: &impl Embedder,
    dest: Destination,
    per_retriever: usize,
    char_budget: usize,
    decay: Option<Decay>,
    weights: RetrievalWeights,
) -> Vec<BundleItem> {
    let query_vec = embedder.embed(query);
    let vector_rank: Vec<MemoryId> = index
        .search(&query_vec, per_retriever)
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    fuse_and_pack(
        query,
        memories,
        &vector_rank,
        dest,
        per_retriever,
        char_budget,
        decay,
        weights,
    )
}

/// The index-agnostic core of the read path: given a pre-computed `vector_rank` (from
/// *any* index — the exact [`VectorIndex`] or the approximate [`IvfIndex`]), fuse it
/// with the recency and keyword retrievers, apply the optional forgetting curve, and
/// pack the result through the egress filter. Every context assembler funnels through
/// the one `pack_bundle` choke point, so ADR-0021 holds regardless of the index.
#[allow(clippy::too_many_arguments)]
pub fn fuse_and_pack(
    query: &str,
    memories: &[Memory],
    vector_rank: &[MemoryId],
    dest: Destination,
    per_retriever: usize,
    char_budget: usize,
    decay: Option<Decay>,
    weights: RetrievalWeights,
) -> Vec<BundleItem> {
    let mut by_recency: Vec<&Memory> = memories.iter().collect();
    by_recency.sort_by_key(|b| std::cmp::Reverse(b.at));
    let recency_rank: Vec<MemoryId> = by_recency
        .iter()
        .take(per_retriever)
        .map(|m| m.id)
        .collect();

    let keyword: Vec<MemoryId> = bm25_rank(query, memories)
        .into_iter()
        .take(per_retriever)
        .collect();

    // Weighted fusion: the dense/recency/keyword lists each vote with their configured weight.
    let mut fused = rrf_fuse_weighted(&[
        (weights.dense, vector_rank),
        (weights.recency, &recency_rank),
        (weights.keyword, &keyword),
    ]);

    // One id→memory map for the (up to two) resolution passes below, so each is an O(1)
    // lookup instead of an O(fused·N) linear scan over the whole corpus per query.
    let by_id: HashMap<MemoryId, &Memory> = memories.iter().map(|m| (m.id, m)).collect();

    // Optional forgetting curve: scale each fused score by the memory's importance and
    // its recency weight, then re-sort. Recent, important memories rise; stale ones sink.
    if let Some(d) = decay {
        for hit in &mut fused {
            if let Some(m) = by_id.get(&hit.id) {
                let age = d.now.saturating_sub(m.at);
                hit.score = decayed_score(hit.score, m.importance, age, d.half_life);
            }
        }
        fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    }

    // Resolve fused ids back to memories, preserving fused order; ids with no memory
    // (e.g. a since-forgotten one) simply drop out.
    let ordered: Vec<&Memory> = fused
        .iter()
        .filter_map(|f| by_id.get(&f.id).copied())
        .collect();

    // Suppress near-duplicate memories so the budget isn't spent on repeats (diversity),
    // then pack the survivors through the egress choke point.
    let diverse = dedup_similar(&ordered, DEDUP_THRESHOLD);
    super::pack_bundle(&diverse, dest, char_budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EgressTier, MemoryKind};

    fn mem(id: MemoryId, tier: EgressTier, at: u64, content: &str) -> Memory {
        mem_imp(id, tier, at, 1.0, content)
    }

    fn mem_imp(id: MemoryId, tier: EgressTier, at: u64, importance: f32, content: &str) -> Memory {
        Memory {
            id,
            kind: MemoryKind::Episodic,
            tier,
            at,
            importance,
            content: content.to_string(),
            redacted: "[redacted]".to_string(),
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    /// A deterministic embedder: text → `[vowel_count, consonant_count]`.
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

    #[test]
    fn rrf_scores_higher_ranks_higher() {
        let fused = rrf_fuse(&[vec![10, 20, 30]]);
        assert_eq!(
            fused.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
        assert!(approx(fused[0].score, 1.0 / 61.0));
        assert!(approx(fused[1].score, 1.0 / 62.0));
        assert!(fused[0].score > fused[1].score);
    }

    #[test]
    fn rrf_accumulates_and_dedupes_across_lists() {
        // id 1 appears rank-1 in both lists; id 2 only once.
        let fused = rrf_fuse(&[vec![1, 2], vec![1]]);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, 1);
        assert!(approx(fused[0].score, 2.0 / 61.0)); // 1/61 + 1/61
        assert!(approx(fused[1].score, 1.0 / 62.0)); // id 2, rank 2 in list 1
    }

    #[test]
    fn rrf_rewards_agreement_over_single_list_dominance() {
        // id 1 is only ever rank-2, but appears in all three lists; ids 7/8/9 are each
        // rank-1 in exactly one. Agreement (3 × 1/62) must beat a lone top (1/61).
        let fused = rrf_fuse(&[vec![9, 1], vec![8, 1], vec![7, 1]]);
        assert_eq!(fused[0].id, 1);
        assert!(approx(fused[0].score, 3.0 / 62.0));
    }

    #[test]
    fn bm25_term_score_matches_the_hand_computed_formula() {
        // Every operator in the formula is pinned by an exact value:
        // tf=1, df=1, n=2, dl=avgdl=3  →  idf=ln(2), norm=1.0  →  ln(2).
        assert!((bm25_term_score(1.0, 1.0, 2.0, 3.0, 3.0) - 2.0_f32.ln()).abs() < 1e-5);
        // A common term (df=2) earns a smaller IDF: ln(1 + 0.5/2.5) = ln(1.2).
        assert!((bm25_term_score(1.0, 2.0, 2.0, 3.0, 3.0) - 1.2_f32.ln()).abs() < 1e-5);
        // A longer document (dl=6, avgdl=3) is penalized: norm = 2.2 / 3.1.
        let long = bm25_term_score(1.0, 1.0, 2.0, 6.0, 3.0);
        assert!((long - 2.0_f32.ln() * (2.2 / 3.1)).abs() < 1e-5);
        // Repetition raises the score, but sub-linearly (saturation via k1).
        assert!(
            bm25_term_score(3.0, 1.0, 2.0, 3.0, 3.0) > bm25_term_score(1.0, 1.0, 2.0, 3.0, 3.0)
        );
    }

    #[test]
    fn bm25_rank_orders_by_rarity_and_excludes_non_matches() {
        // "the" is in both matches (common → low IDF); "cat" is rare (high IDF). The memory
        // with the rare term outranks the one with only the common term; a memory with
        // neither query term is excluded entirely.
        let mems = vec![
            mem(1, EgressTier::Open, 1, "the dog ran"), // "the" only
            mem(2, EgressTier::Open, 2, "the cat sat"), // "the" + rare "cat"
            mem(3, EgressTier::Open, 3, "a bird flew"), // neither → excluded
        ];
        assert_eq!(bm25_rank("the cat", &mems), vec![2, 1]);
    }

    #[test]
    fn bm25_rank_prefers_the_shorter_of_two_equal_matches() {
        // Both contain "signal" once; length normalization ranks the shorter memory first.
        let mems = vec![
            mem(
                1,
                EgressTier::Open,
                1,
                "signal amid a great deal of extra padding words here",
            ),
            mem(2, EgressTier::Open, 2, "signal short"),
        ];
        assert_eq!(bm25_rank("signal", &mems), vec![2, 1]);
    }

    #[test]
    fn bm25_rank_matches_case_insensitively() {
        let mems = vec![mem(1, EgressTier::Open, 1, "The CAT, sat!")];
        assert_eq!(bm25_rank("cat", &mems), vec![1]);
    }

    #[test]
    fn bm25_rank_is_empty_for_no_query_terms_or_no_memories() {
        let mems = vec![mem(1, EgressTier::Open, 1, "a stored memory")];
        assert!(bm25_rank("   ", &mems).is_empty()); // no query terms
        assert!(bm25_rank("memory", &[]).is_empty()); // no memories
    }

    #[test]
    fn avg_len_is_the_mean_document_length_floored_at_one() {
        assert_eq!(avg_len(12, 2.0), 6.0);
        assert_eq!(avg_len(0, 3.0), 1.0); // floored to 1.0, never 0 (keeps the length term finite)
    }

    #[test]
    fn content_similarity_is_token_jaccard() {
        assert_eq!(content_similarity("a b c", "a b c"), 1.0); // identical
        assert_eq!(content_similarity("a b c", "x y z"), 0.0); // disjoint
        assert_eq!(content_similarity("a b c", "b c d"), 0.5); // {b,c} / {a,b,c,d}
        assert_eq!(content_similarity("", ""), 1.0); // both empty → identical
        assert_eq!(content_similarity("", "a"), 0.0); // one empty
    }

    #[test]
    fn dedup_similar_drops_near_duplicates_keeping_the_first() {
        let a = mem(1, EgressTier::Open, 3, "alpha beta gamma delta epsilon");
        let b = mem(2, EgressTier::Open, 2, "epsilon delta gamma beta alpha"); // same token set
        let c = mem(3, EgressTier::Open, 1, "one two three four five"); // distinct
        let ordered = vec![&a, &b, &c];
        let kept: Vec<MemoryId> = dedup_similar(&ordered, DEDUP_THRESHOLD)
            .iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(kept, vec![1, 3]); // 2 is a near-duplicate of 1; 3 survives

        // A threshold above 1.0 keeps everything.
        let all: Vec<MemoryId> = dedup_similar(&ordered, 1.01).iter().map(|m| m.id).collect();
        assert_eq!(all, vec![1, 2, 3]);

        // At an exact-1.0 threshold, identical content IS a duplicate (`>=`, not `>`).
        let d = mem(4, EgressTier::Open, 1, "same words here");
        let e = mem(5, EgressTier::Open, 1, "same words here");
        let kept2: Vec<MemoryId> = dedup_similar(&[&d, &e], 1.0).iter().map(|m| m.id).collect();
        assert_eq!(kept2, vec![4]);
    }

    #[test]
    fn hybrid_recall_fuses_and_honours_the_budget() {
        let mems = vec![
            mem(1, EgressTier::Open, 1, "the cat sat"),
            mem(2, EgressTier::Open, 2, "the dog ran"),
        ];
        let mut idx = VectorIndex::new(2);
        let embedder = VowelEmbedder;
        idx.insert(1, embedder.embed("the cat sat")).unwrap();
        idx.insert(2, embedder.embed("the dog ran")).unwrap();

        let bundle = hybrid_recall(
            "dog",
            &mems,
            &idx,
            &embedder,
            Destination::Local,
            10,
            1_000,
            None,
            RetrievalWeights::default(),
        );
        // "dog" wins on both the keyword and recency retrievers, so memory 2 must be
        // the TOP hit — and its item must carry memory 2's own content (a mis-resolved
        // id↔memory lookup would swap the order or the text).
        assert_eq!(bundle.len(), 2); // both fit the generous budget
        assert_eq!(bundle[0].id, 2);
        assert_eq!(bundle[0].text, "the dog ran");
        assert!(bundle.iter().any(|b| b.id == 1 && b.text == "the cat sat"));
    }

    #[test]
    fn hybrid_recall_never_leaks_a_private_memory_to_a_remote_bundle() {
        let mems = vec![
            mem(1, EgressTier::Open, 1, "public note"),
            mem(2, EgressTier::Private, 2, "the secret dog plan"),
        ];
        let mut idx = VectorIndex::new(2);
        let embedder = VowelEmbedder;
        idx.insert(1, embedder.embed("public note")).unwrap();
        idx.insert(2, embedder.embed("the secret dog plan"))
            .unwrap();

        // Even though "dog" ranks the private memory highly, a Remote bundle drops it.
        let remote = hybrid_recall(
            "dog",
            &mems,
            &idx,
            &embedder,
            Destination::Remote,
            10,
            1_000,
            None,
            RetrievalWeights::default(),
        );
        assert!(remote.iter().all(|b| b.id != 2));
        assert!(remote.iter().all(|b| !b.text.contains("secret")));

        // ...but a Local bundle may include it (full content).
        let local = hybrid_recall(
            "dog",
            &mems,
            &idx,
            &embedder,
            Destination::Local,
            10,
            1_000,
            None,
            RetrievalWeights::default(),
        );
        assert!(
            local
                .iter()
                .any(|b| b.id == 2 && b.text == "the secret dog plan")
        );
    }

    #[test]
    fn decay_weight_halves_each_half_life() {
        assert!(approx(decay_weight(0, 10), 1.0)); // brand new → full weight
        assert!(approx(decay_weight(10, 10), 0.5)); // one half-life
        assert!(approx(decay_weight(20, 10), 0.25)); // two half-lives
        assert!(approx(decay_weight(7, 0), 1.0)); // half_life 0 disables decay
    }

    #[test]
    fn decay_lifts_a_recent_memory_over_an_equally_relevant_stale_one() {
        // Identical content, so vector + keyword are symmetric ties; only age differs.
        let mems = vec![
            mem(1, EgressTier::Open, 1, "alpha beta"),   // old
            mem(2, EgressTier::Open, 100, "alpha beta"), // recent
        ];
        let mut idx = VectorIndex::new(2);
        let embedder = VowelEmbedder;
        idx.insert(1, embedder.embed("alpha beta")).unwrap();
        idx.insert(2, embedder.embed("alpha beta")).unwrap();

        let decay = Some(Decay {
            now: 100,
            half_life: 10,
        });
        let bundle = hybrid_recall(
            "alpha",
            &mems,
            &idx,
            &embedder,
            Destination::Local,
            10,
            1_000,
            decay,
            RetrievalWeights::default(),
        );
        // Memory 1 is ~99 ticks old (weight ≈ 0.5^9.9); memory 2 is fresh — it wins.
        assert_eq!(bundle[0].id, 2);
    }

    #[test]
    fn importance_lifts_a_salient_memory_over_a_neutral_one() {
        // Same content and same age, so decay and base scores are symmetric; only the
        // importance multiplier differs.
        let mems = vec![
            mem_imp(1, EgressTier::Open, 5, 1.0, "alpha beta"), // neutral
            mem_imp(2, EgressTier::Open, 5, 10.0, "alpha beta"), // salient
        ];
        let mut idx = VectorIndex::new(2);
        let embedder = VowelEmbedder;
        idx.insert(1, embedder.embed("alpha beta")).unwrap();
        idx.insert(2, embedder.embed("alpha beta")).unwrap();

        let decay = Some(Decay {
            now: 5,
            half_life: 10,
        });
        let bundle = hybrid_recall(
            "alpha",
            &mems,
            &idx,
            &embedder,
            Destination::Local,
            10,
            1_000,
            decay,
            RetrievalWeights::default(),
        );
        // 10× importance overwhelms memory 1's slight stable-tie edge.
        assert_eq!(bundle[0].id, 2);
    }

    #[test]
    fn decayed_score_is_the_product_of_base_importance_and_weight() {
        // The forgetting curve is MULTIPLICATIVE. At age 0 the weight is 1.0, so the score
        // is base × importance = 6.0; an additive combination would give 2 + 3 = 5.0.
        assert!(approx(decayed_score(2.0, 3.0, 0, 10), 6.0));
        // At one half-life the weight halves: 4 × 1 × 0.5 = 2.0.
        assert!(approx(decayed_score(4.0, 1.0, 10, 10), 2.0));
        // Importance multiplies too: 1 × 3 × 0.5 = 1.5.
        assert!(approx(decayed_score(1.0, 3.0, 10, 10), 1.5));
    }

    #[test]
    fn the_dense_retriever_actually_feeds_the_fusion() {
        // mem 1 matches the query ONLY by embedding: "iou" and the query "aei" share no
        // token (so keyword = 0) and mem 1 is older than mem 2 (so recency, capped at 1,
        // surfaces mem 2, not mem 1). Both embed to [3, 0], so cosine picks mem 1. If it
        // reaches the bundle, the dense retriever fed the fusion — dropping vector_rank
        // would make it vanish.
        let mems = vec![
            mem(1, EgressTier::Open, 1, "iou"),
            mem(2, EgressTier::Open, 2, "bcd"),
        ];
        let embedder = VowelEmbedder;
        let mut idx = VectorIndex::new(2);
        idx.insert(1, embedder.embed("iou")).unwrap();
        idx.insert(2, embedder.embed("bcd")).unwrap();
        let bundle = hybrid_recall(
            "aei",
            &mems,
            &idx,
            &embedder,
            Destination::Local,
            1,
            1_000,
            None,
            RetrievalWeights::default(),
        );
        assert!(
            bundle.iter().any(|b| b.id == 1),
            "the embedding-only match must be recalled via the dense retriever: {bundle:?}"
        );
    }

    #[test]
    fn weighting_the_dense_retriever_flips_a_lexical_recency_decoy() {
        // The exact shape that defeats balanced fusion: memory 1 is the semantic (dense) match;
        // memory 2 merely shares a keyword AND is the most recent, so under equal weights its
        // two votes beat memory 1's one.
        let dense = vec![1u64];
        let keyword = vec![2u64];
        let recency = vec![2u64];

        let equal = rrf_fuse_weighted(&[(1.0, &dense), (1.0, &keyword), (1.0, &recency)]);
        assert_eq!(
            equal[0].id, 2,
            "balanced fusion: the lexical + recent decoy wins"
        );

        // Weight the dense retriever up (RetrievalWeights::semantic) and the single semantic
        // vote outweighs the decoy's two — the meaning-match now wins.
        let w = RetrievalWeights::semantic();
        let weighted = rrf_fuse_weighted(&[
            (w.dense, &dense),
            (w.keyword, &keyword),
            (w.recency, &recency),
        ]);
        assert_eq!(
            weighted[0].id, 1,
            "dense-weighted fusion: the semantic match wins"
        );
    }

    #[test]
    fn default_weights_leave_recall_unchanged() {
        // recall_weighted with Default weights must fuse identically to plain rrf_fuse — the
        // weighting is purely additive on top of today's behavior.
        let a = rrf_fuse(&[vec![1, 2], vec![2, 3]]);
        let w = RetrievalWeights::default();
        let b = rrf_fuse_weighted(&[(w.dense, &[1, 2]), (w.recency, &[2, 3]), (w.keyword, &[])]);
        assert_eq!(
            a.iter().map(|f| f.id).collect::<Vec<_>>(),
            b.iter().map(|f| f.id).collect::<Vec<_>>()
        );
    }
}
