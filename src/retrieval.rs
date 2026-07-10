//! Hybrid retrieval — Phase-2 slice B (`docs/proposals/engram-memory-layer.md` §3.3).
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
    let mut acc: Vec<Fused> = Vec::new();
    for list in rankings {
        for (pos, id) in list.iter().enumerate() {
            let rank = pos as f32 + 1.0;
            let contribution = 1.0 / (RRF_K + rank);
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

/// Rank memories by descending count of distinct query terms found in their content
/// (a cheap lexical signal). Only memories with at least one shared term are returned;
/// ties keep input order (the sort is stable).
pub fn keyword_rank(query: &str, memories: &[Memory]) -> Vec<MemoryId> {
    let terms = tokenize(query);
    let mut scored: Vec<(MemoryId, usize)> = memories
        .iter()
        .filter_map(|m| {
            let content = tokenize(&m.content);
            let overlap = terms.iter().filter(|t| content.contains(t)).count();
            if overlap > 0 {
                Some((m.id, overlap))
            } else {
                None
            }
        })
        .collect();
    scored.sort_by_key(|b| std::cmp::Reverse(b.1));
    scored.into_iter().map(|(id, _)| id).collect()
}

/// Lowercase, split on non-alphanumeric, drop empties. Distinct-preserving is not
/// needed — `keyword_rank` counts distinct query terms, and duplicates in a term list
/// only re-confirm a match.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
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
) -> Vec<BundleItem> {
    let mut by_recency: Vec<&Memory> = memories.iter().collect();
    by_recency.sort_by_key(|b| std::cmp::Reverse(b.at));
    let recency_rank: Vec<MemoryId> = by_recency
        .iter()
        .take(per_retriever)
        .map(|m| m.id)
        .collect();

    let keyword: Vec<MemoryId> = keyword_rank(query, memories)
        .into_iter()
        .take(per_retriever)
        .collect();

    let mut fused = rrf_fuse(&[vector_rank.to_vec(), recency_rank, keyword]);

    // Optional forgetting curve: scale each fused score by the memory's importance and
    // its recency weight, then re-sort. Recent, important memories rise; stale ones sink.
    if let Some(d) = decay {
        for hit in &mut fused {
            if let Some(m) = memories.iter().find(|m| m.id == hit.id) {
                let age = d.now.saturating_sub(m.at);
                hit.score *= m.importance * decay_weight(age, d.half_life);
            }
        }
        fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    }

    // Resolve fused ids back to memories, preserving fused order; ids with no memory
    // (e.g. a since-forgotten one) simply drop out.
    let ordered: Vec<&Memory> = fused
        .iter()
        .filter_map(|f| memories.iter().find(|m| m.id == f.id))
        .collect();

    super::pack_bundle(&ordered, dest, char_budget)
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
    fn keyword_rank_orders_by_overlap_and_excludes_non_matches() {
        let mems = vec![
            mem(1, EgressTier::Open, 1, "the cat and the dog"), // matches cat, dog → 2
            mem(2, EgressTier::Open, 2, "the dog ran"),         // matches dog → 1
            mem(3, EgressTier::Open, 3, "a bird flew"),         // no overlap → excluded
        ];
        let ranked = keyword_rank("cat dog", &mems);
        assert_eq!(ranked, vec![1, 2]); // 3 is absent; 1 outranks 2 by overlap
    }

    #[test]
    fn keyword_rank_is_case_insensitive_and_tokenized() {
        let mems = vec![mem(1, EgressTier::Open, 1, "The CAT, sat!")];
        assert_eq!(keyword_rank("cat", &mems), vec![1]);
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
        );
        // 10× importance overwhelms memory 1's slight stable-tie edge.
        assert_eq!(bundle[0].id, 2);
    }
}
