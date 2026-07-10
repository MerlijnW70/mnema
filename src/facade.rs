//! The `Engram` facade — Phase-2 slice C (`docs/proposals/engram-memory-layer.md`).
//! One object that turns the proven parts into a usable memory layer: an encrypted
//! episodic store (slice 2), a contradiction-resolving semantic store (slice 3), an
//! exact vector index (slice 2a), and hybrid retrieval (slice 2b), coordinated behind
//! a small `remember` / `recall` / `forget` API.
//!
//! The facade owns id assignment and a monotonic logical clock, so a caller never has
//! to supply a timestamp or keep the vector index in sync by hand — `remember` appends
//! the event, embeds it, and indexes it under the same id; `forget` hard-deletes from
//! both the log and the index in one call.
//!
//! Gated behind the `secure` feature (ADR-0020) because it builds on the encrypted
//! episodic store. The retrieval and semantic primitives remain available without it.

use crate::retrieval::{Decay, fuse_and_pack, hybrid_recall};
use crate::semantic::{Fact, FactStatus, Resolution, SemanticStore};
use crate::store::{
    EpisodicLog, PurgeReceipt, StoreError, open_bytes, put_bytes, seal_bytes, string_from,
    take_bytes, take_u8, take_u32, take_u64,
};
use crate::vector::{Embedder, IvfIndex, VectorIndex};
use crate::working::{Note, WorkingMemory};
use crate::{BundleItem, Destination, EgressTier, Memory, MemoryId, MemoryKind};

/// Default scratchpad TTL: a note stays live for this many `remember`/`scratch` ticks.
const WORKING_HORIZON: u64 = 32;
/// Default scratchpad capacity: at most this many notes at once.
const WORKING_CAPACITY: usize = 16;

/// A batteries-included local-first memory layer over a pluggable [`Embedder`].
pub struct Engram<E: Embedder> {
    episodic: EpisodicLog,
    index: VectorIndex,
    /// An optional approximate index for fast recall, built on demand by [`build_ann`].
    /// The exact `index` remains the source of truth and the default recall path.
    ///
    /// [`build_ann`]: Engram::build_ann
    ann: Option<IvfIndex>,
    semantic: SemanticStore,
    working: WorkingMemory,
    embedder: E,
    clock: u64,
}

impl<E: Embedder> Engram<E> {
    /// A new, empty memory over `embedder` (whose `dims()` fixes the index width).
    pub fn new(embedder: E) -> Self {
        let dims = embedder.dims();
        Self {
            episodic: EpisodicLog::new(),
            index: VectorIndex::new(dims),
            ann: None,
            semantic: SemanticStore::new(),
            working: WorkingMemory::new(WORKING_HORIZON, WORKING_CAPACITY),
            embedder,
            clock: 0,
        }
    }

    /// The next logical timestamp — a monotonic counter so later memories are more
    /// recent without the caller tracking a clock.
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Store an episodic memory of neutral importance; append, embed, index under one id.
    pub fn remember(&mut self, tier: EgressTier, content: &str) -> MemoryId {
        self.remember_important(tier, 1.0, content)
    }

    /// Store an episodic memory with an explicit `importance` (see [`Memory::importance`]),
    /// which lifts it against the forgetting curve in [`recall_decayed`](Engram::recall_decayed).
    pub fn remember_important(
        &mut self,
        tier: EgressTier,
        importance: f32,
        content: &str,
    ) -> MemoryId {
        let at = self.tick();
        let vector = self.embedder.embed(content);
        let id =
            self.episodic
                .append_important(MemoryKind::Episodic, tier, at, importance, content);
        // The index width equals the embedder's dims, so a well-behaved embedder never
        // mismatches; a broken one just leaves this memory unindexed (still stored).
        let _ = self.index.insert(id, vector);
        id
    }

    /// Assert a semantic fact, resolving contradictions ([`SemanticStore::assert`]).
    pub fn remember_fact(&mut self, subject: &str, attribute: &str, value: &str) -> Resolution {
        let at = self.tick();
        self.semantic.assert(subject, attribute, value, at)
    }

    /// The current belief for a `(subject, attribute)` key, if any.
    pub fn belief(&self, subject: &str, attribute: &str) -> Option<&Fact> {
        self.semantic.current(subject, attribute)
    }

    /// Hybrid recall over the episodic memories, egress-filtered for `dest` and bounded
    /// to `char_budget` characters. `per_retriever` caps each retriever's shortlist.
    pub fn recall(
        &self,
        query: &str,
        dest: Destination,
        per_retriever: usize,
        char_budget: usize,
    ) -> Vec<BundleItem> {
        hybrid_recall(
            query,
            self.episodic.events(),
            &self.index,
            &self.embedder,
            dest,
            per_retriever,
            char_budget,
            None,
        )
    }

    /// Like [`recall`](Engram::recall), but applies the forgetting curve (proposal §3.2):
    /// each hit's score is scaled by its `importance` and a recency weight that halves
    /// every `half_life` ticks, using the facade's own clock as *now*. Recent, important
    /// memories are preferred over stale ones; a `half_life` of `0` behaves like `recall`.
    pub fn recall_decayed(
        &self,
        query: &str,
        dest: Destination,
        per_retriever: usize,
        char_budget: usize,
        half_life: u64,
    ) -> Vec<BundleItem> {
        hybrid_recall(
            query,
            self.episodic.events(),
            &self.index,
            &self.embedder,
            dest,
            per_retriever,
            char_budget,
            Some(Decay {
                now: self.clock,
                half_life,
            }),
        )
    }

    /// Build the approximate (IVF) index over the current corpus, enabling
    /// [`recall_fast`](Engram::recall_fast). `num_anchors` buckets are seeded from the
    /// first `num_anchors` memories' embeddings (a deterministic placeholder for a
    /// future k-means); every memory is then bucketed. Call again to rebuild after the
    /// corpus changes — the exact index stays the source of truth either way.
    pub fn build_ann(&mut self, num_anchors: usize) {
        let vectors: Vec<(MemoryId, Vec<f32>)> = self
            .episodic
            .events()
            .iter()
            .map(|e| (e.id, self.embedder.embed(&e.content)))
            .collect();
        let anchors: Vec<Vec<f32>> = vectors
            .iter()
            .take(num_anchors)
            .map(|(_, v)| v.clone())
            .collect();
        let mut ann = IvfIndex::new(self.embedder.dims(), anchors);
        for (id, v) in vectors {
            let _ = ann.insert(id, v);
        }
        self.ann = Some(ann);
    }

    /// Like [`recall`](Engram::recall), but sources the dense retriever from the
    /// approximate index (scanning `probe` buckets) when one has been built with
    /// [`build_ann`] — trading recall for speed. Without a built index it falls back to
    /// the exact path, so the result is always egress-safe and never worse than exact.
    pub fn recall_fast(
        &self,
        query: &str,
        dest: Destination,
        per_retriever: usize,
        char_budget: usize,
        probe: usize,
    ) -> Vec<BundleItem> {
        let query_vec = self.embedder.embed(query);
        let vector_rank: Vec<MemoryId> = match &self.ann {
            Some(ann) => ann
                .search(&query_vec, per_retriever, probe)
                .into_iter()
                .map(|hit| hit.id)
                .collect(),
            None => self
                .index
                .search(&query_vec, per_retriever)
                .into_iter()
                .map(|hit| hit.id)
                .collect(),
        };
        fuse_and_pack(
            query,
            self.episodic.events(),
            &vector_rank,
            dest,
            per_retriever,
            char_budget,
            None,
        )
    }

    /// The `k` most-recent memories by logical time, newest first.
    pub fn recall_by_recency(&self, k: usize) -> Vec<&Memory> {
        let mut ordered: Vec<&Memory> = self.episodic.events().iter().collect();
        ordered.sort_by_key(|b| std::cmp::Reverse(b.at));
        ordered.truncate(k);
        ordered
    }

    /// Hard-delete matching memories from both the log and the vector index, returning
    /// the episodic [`PurgeReceipt`]. Forgotten ids are never reused (see `EpisodicLog`).
    pub fn forget(&mut self, predicate: impl FnMut(&Memory) -> bool) -> PurgeReceipt {
        let receipt = self.episodic.forget(predicate);
        for id in &receipt.purged {
            self.index.remove(*id);
        }
        receipt
    }

    /// Number of stored episodic memories.
    pub fn len(&self) -> usize {
        self.episodic.len()
    }

    /// Whether no episodic memories are stored.
    pub fn is_empty(&self) -> bool {
        self.episodic.is_empty()
    }

    /// Number of vectors currently in the index (kept in step with `len` by `forget`).
    pub fn indexed(&self) -> usize {
        self.index.len()
    }

    /// Encrypt the whole memory at rest — the episodic log, the semantic beliefs, and
    /// the logical clock — as one blob (`salt || nonce || AEAD(...)`). The vector index
    /// is *derived* (rebuildable by re-embedding), so it is not stored; [`open`]
    /// reconstructs it. Sealing routes through the same AEAD as the raw store.
    ///
    /// [`open`]: Engram::open
    pub fn seal(&self, passphrase: &[u8]) -> Result<Vec<u8>, StoreError> {
        let mut plain = Vec::new();
        plain.extend_from_slice(&self.clock.to_le_bytes());
        put_bytes(&mut plain, &self.episodic.encode());
        put_bytes(&mut plain, &encode_facts(self.semantic.facts()));
        seal_bytes(&plain, passphrase)
    }

    /// Recover a whole memory from a [`seal`](Engram::seal)ed blob with `passphrase` and
    /// an `embedder` (whose `dims()` must match the one that sealed it). The episodic log,
    /// beliefs, and clock are restored verbatim; the vector index is rebuilt by
    /// re-embedding every event, so recall resumes immediately. A wrong key or tampering
    /// yields [`StoreError::Decrypt`].
    pub fn open(blob: &[u8], passphrase: &[u8], embedder: E) -> Result<Self, StoreError> {
        let plain = open_bytes(blob, passphrase)?;
        let (clock, off) = take_u64(&plain, 0)?;
        let (episodic_bytes, off) = take_bytes(&plain, off)?;
        let (semantic_bytes, _off) = take_bytes(&plain, off)?;

        let episodic = EpisodicLog::decode(episodic_bytes)?;
        let semantic = SemanticStore::from_facts(decode_facts(semantic_bytes)?);

        // The index is derived: rebuild it from the events (same id space).
        let mut index = VectorIndex::new(embedder.dims());
        for e in episodic.events() {
            let _ = index.insert(e.id, embedder.embed(&e.content));
        }

        Ok(Self {
            episodic,
            index,
            ann: None, // derived; rebuild explicitly with `build_ann` after opening
            semantic,
            // Working memory is ephemeral — it is not sealed, so `open` starts fresh.
            working: WorkingMemory::new(WORKING_HORIZON, WORKING_CAPACITY),
            embedder,
            clock,
        })
    }

    /// Write to the ephemeral scratchpad (working memory): a short-lived note stamped
    /// at the current tick. Notes expire after [`WORKING_HORIZON`] ticks and are not
    /// persisted by [`seal`](Engram::seal).
    pub fn scratch(&mut self, content: &str) {
        let at = self.tick();
        self.working.note(at, content);
    }

    /// The live scratchpad notes as of now, newest first.
    pub fn scratchpad(&self) -> Vec<&Note> {
        self.working.active(self.clock)
    }
}

fn status_tag(status: FactStatus) -> u8 {
    match status {
        FactStatus::Live => 0,
        FactStatus::Superseded => 1,
    }
}

fn status_from_tag(tag: u8) -> Result<FactStatus, StoreError> {
    match tag {
        0 => Ok(FactStatus::Live),
        1 => Ok(FactStatus::Superseded),
        _ => Err(StoreError::UnknownTag),
    }
}

/// Serialize semantic facts: per fact `id(8) | at(8) | confidence(4) | status(1)` then
/// the length-prefixed subject, attribute, and value.
fn encode_facts(facts: &[Fact]) -> Vec<u8> {
    let mut buf = Vec::new();
    for f in facts {
        buf.extend_from_slice(&f.id.to_le_bytes());
        buf.extend_from_slice(&f.at.to_le_bytes());
        buf.extend_from_slice(&f.confidence.to_le_bytes());
        buf.push(status_tag(f.status));
        put_bytes(&mut buf, f.subject.as_bytes());
        put_bytes(&mut buf, f.attribute.as_bytes());
        put_bytes(&mut buf, f.value.as_bytes());
    }
    buf
}

/// Parse the fact wire format back. Every read is bounds-checked ([`take_u64`] etc.),
/// so a truncated blob yields [`StoreError::Truncated`], never a panic.
fn decode_facts(buf: &[u8]) -> Result<Vec<Fact>, StoreError> {
    let mut facts = Vec::new();
    let mut off = 0usize;
    while off < buf.len() {
        let (id, o) = take_u64(buf, off)?;
        let (at, o) = take_u64(buf, o)?;
        let (confidence, o) = take_u32(buf, o)?;
        let (status_byte, o) = take_u8(buf, o)?;
        let (subject, o) = take_bytes(buf, o)?;
        let (attribute, o) = take_bytes(buf, o)?;
        let (value, next) = take_bytes(buf, o)?;
        off = next;
        facts.push(Fact {
            id,
            subject: string_from(subject)?,
            attribute: string_from(attribute)?,
            value: string_from(value)?,
            at,
            confidence,
            status: status_from_tag(status_byte)?,
        });
    }
    Ok(facts)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn remember_assigns_ids_and_counts_both_stores() {
        let mut e = Engram::new(VowelEmbedder);
        assert_eq!(e.remember(EgressTier::Open, "the cat sat"), 0);
        assert_eq!(e.remember(EgressTier::Open, "the dog ran"), 1);
        assert_eq!(e.len(), 2);
        assert_eq!(e.indexed(), 2);
        assert!(!e.is_empty());
    }

    #[test]
    fn recall_surfaces_the_keyword_match_first() {
        let mut e = Engram::new(VowelEmbedder);
        e.remember(EgressTier::Open, "the cat sat"); // id 0
        e.remember(EgressTier::Open, "the dog ran"); // id 1
        let bundle = e.recall("dog", Destination::Local, 10, 1_000);
        assert_eq!(bundle[0].id, 1);
        assert_eq!(bundle[0].text, "the dog ran");
    }

    #[test]
    fn recall_respects_egress() {
        let mut e = Engram::new(VowelEmbedder);
        e.remember(EgressTier::Private, "the secret dog plan"); // id 0
        e.remember(EgressTier::Open, "a public cat note"); // id 1
        let remote = e.recall("dog", Destination::Remote, 10, 1_000);
        assert!(remote.iter().all(|b| !b.text.contains("secret")));
        let local = e.recall("dog", Destination::Local, 10, 1_000);
        assert!(local.iter().any(|b| b.text == "the secret dog plan"));
    }

    #[test]
    fn forget_removes_from_both_recall_and_index() {
        let mut e = Engram::new(VowelEmbedder);
        e.remember(EgressTier::Open, "the cat sat"); // 0
        e.remember(EgressTier::Open, "the dog ran"); // 1
        e.remember(EgressTier::Open, "the bird flew"); // 2
        let receipt = e.forget(|m| m.content.contains("dog"));
        assert_eq!(receipt.purged, vec![1]);
        assert_eq!(e.len(), 2);
        assert_eq!(e.indexed(), 2); // index stayed in step with the log
        // The forgotten memory can no longer be recalled.
        let hits = e.recall("dog", Destination::Local, 10, 1_000);
        assert!(hits.iter().all(|b| !b.text.contains("dog")));
    }

    #[test]
    fn newer_memories_are_more_recent() {
        let mut e = Engram::new(VowelEmbedder);
        let a = e.remember(EgressTier::Open, "alpha");
        let b = e.remember(EgressTier::Open, "bravo");
        let recent = e.recall_by_recency(2);
        // Pins the monotonic clock: a stalled/rewinding clock would flip this order.
        assert_eq!(recent[0].id, b);
        assert_eq!(recent[1].id, a);
    }

    #[test]
    fn facts_resolve_contradictions_and_belief_reads_the_current_one() {
        let mut e = Engram::new(VowelEmbedder);
        assert_eq!(
            e.remember_fact("alice", "diet", "vegetarian"),
            Resolution::Inserted
        );
        assert_eq!(
            e.remember_fact("alice", "diet", "omnivore"),
            Resolution::Superseded
        );
        assert_eq!(
            e.belief("alice", "diet").map(|f| f.value.as_str()),
            Some("omnivore")
        );
    }

    #[test]
    fn the_scratchpad_holds_ephemeral_notes_newest_first() {
        let mut e = Engram::new(VowelEmbedder);
        e.scratch("first thought");
        e.scratch("second thought");
        let pad: Vec<&str> = e.scratchpad().iter().map(|n| n.content.as_str()).collect();
        assert_eq!(pad, vec!["second thought", "first thought"]);
        // Scratch notes are working memory, not episodic — they don't count as memories.
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn recall_fast_falls_back_to_exact_without_a_built_index() {
        let mut e = Engram::new(VowelEmbedder);
        e.remember(EgressTier::Open, "the cat sat");
        e.remember(EgressTier::Open, "the dog ran");
        // No ANN built → recall_fast must equal the exact recall, bundle for bundle.
        let fast = e.recall_fast("dog", Destination::Local, 10, 1_000, 8);
        let exact = e.recall("dog", Destination::Local, 10, 1_000);
        assert_eq!(fast, exact);
    }

    #[test]
    fn recall_fast_at_full_probe_matches_exact_recall() {
        let mut e = Engram::new(VowelEmbedder);
        e.remember(EgressTier::Private, "the secret dog plan");
        e.remember(EgressTier::Open, "a public cat note");
        e.build_ann(2);
        // probe >= anchors → the IVF scans every bucket → identical to exact recall,
        // egress and all (a Remote bundle still drops the private memory).
        let fast = e.recall_fast("dog", Destination::Remote, 10, 1_000, 2);
        let exact = e.recall("dog", Destination::Remote, 10, 1_000);
        assert_eq!(fast, exact);
        assert!(fast.iter().all(|b| !b.text.contains("secret")));
    }

    #[test]
    fn recall_decayed_prefers_the_recent_memory() {
        let mut e = Engram::new(VowelEmbedder);
        e.remember(EgressTier::Open, "alpha beta"); // id 0, at 1 (old)
        e.remember(EgressTier::Open, "alpha beta"); // id 1, at 2 (recent)
        // Same content → symmetric base scores; a short half-life lets recency decide.
        let bundle = e.recall_decayed("alpha", Destination::Local, 10, 1_000, 1);
        assert_eq!(bundle[0].id, 1);
    }

    #[test]
    fn remember_important_resists_the_forgetting_curve() {
        let mut e = Engram::new(VowelEmbedder);
        e.remember_important(EgressTier::Open, 10.0, "alpha beta"); // id 0, older but salient
        e.remember(EgressTier::Open, "alpha beta"); // id 1, newer but neutral
        // A long half-life makes decay negligible, so 10× importance keeps the older,
        // salient memory on top despite the newer one.
        let bundle = e.recall_decayed("alpha", Destination::Local, 10, 1_000, 1_000_000);
        assert_eq!(bundle[0].id, 0);
    }

    fn populated() -> Engram<VowelEmbedder> {
        let mut e = Engram::new(VowelEmbedder);
        e.remember(EgressTier::Open, "the cat sat"); // id 0
        e.remember(EgressTier::Private, "the secret dog plan"); // id 1
        e.remember_fact("alice", "diet", "vegetarian"); // superseded ↓
        e.remember_fact("alice", "diet", "omnivore"); // live
        e
    }

    #[test]
    fn seal_then_open_restores_memories_beliefs_index_and_clock() {
        let sealed = populated().seal(b"key").unwrap();
        let mut reopened = Engram::open(&sealed, b"key", VowelEmbedder).unwrap();

        // Episodic memories and the (rebuilt) index survive.
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.indexed(), 2);
        assert_eq!(
            reopened.recall("dog", Destination::Local, 10, 1_000)[0].id,
            1
        );

        // The belief survives with the RIGHT live value — a mis-decoded status byte
        // would leave the superseded "vegetarian" live and this would read wrong.
        assert_eq!(
            reopened.belief("alice", "diet").map(|f| f.value.as_str()),
            Some("omnivore")
        );

        // The clock resumed: a new memory is more recent than the restored ones, and
        // takes the next episodic id (2), never a reused one.
        let fresh = reopened.remember(EgressTier::Open, "a new thing");
        assert_eq!(fresh, 2);
        assert_eq!(reopened.recall_by_recency(1)[0].id, 2);
    }

    #[test]
    fn a_sealed_memory_is_opaque_at_rest() {
        let sealed = populated().seal(b"key").unwrap();
        // Neither episodic content nor a belief value appears in the ciphertext.
        assert!(!sealed.windows(6).any(|w| w == b"secret"));
        assert!(!sealed.windows(8).any(|w| w == b"omnivore"));
    }

    #[test]
    fn opening_with_the_wrong_passphrase_fails() {
        let sealed = populated().seal(b"right").unwrap();
        assert_eq!(
            Engram::open(&sealed, b"wrong", VowelEmbedder).err(),
            Some(StoreError::Decrypt)
        );
    }

    #[test]
    fn opening_a_truncated_blob_fails() {
        assert_eq!(
            Engram::open(&[0u8; 8], b"key", VowelEmbedder).err(),
            Some(StoreError::Truncated)
        );
    }
}
