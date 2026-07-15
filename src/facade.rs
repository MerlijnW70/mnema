//! The `Mnema` facade — Phase-2 slice C (`docs/proposals/mnema-memory-layer.md`).
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

use crate::retrieval::{Decay, RetrievalWeights, fuse_and_pack, hybrid_recall, is_faded};
use crate::semantic::{Fact, FactStatus, Resolution, SemanticStore};
use crate::store::{
    EpisodicLog, PurgeReceipt, SealingKey, StoreError, put_bytes, string_from, take_bytes, take_u8,
    take_u32, take_u64,
};
use crate::vector::{Embedder, IvfIndex, VectorIndex};
use crate::working::{Note, WorkingMemory};
use crate::{BundleItem, Destination, EgressTier, Memory, MemoryId, MemoryKind};

/// Default scratchpad TTL: a note stays live for this many `remember`/`scratch` ticks.
const WORKING_HORIZON: u64 = 32;
/// Default scratchpad capacity: at most this many notes at once.
const WORKING_CAPACITY: usize = 16;
/// Lloyd iterations for [`build_ann`](Mnema::build_ann)'s k-means anchor training — enough to
/// settle the centroids on typical corpora; bounded so building the index stays fast and O(1) in
/// rounds regardless of corpus size.
const ANN_KMEANS_ITERS: usize = 10;

/// A batteries-included local-first memory layer over a pluggable [`Embedder`].
pub struct Mnema<E: Embedder> {
    episodic: EpisodicLog,
    index: VectorIndex,
    /// An optional approximate index for fast recall, built on demand by [`build_ann`].
    /// The exact `index` remains the source of truth and the default recall path.
    ///
    /// [`build_ann`]: Mnema::build_ann
    ann: Option<IvfIndex>,
    semantic: SemanticStore,
    working: WorkingMemory,
    embedder: E,
    clock: u64,
    /// The derived sealing key, cached after the first [`seal`](Mnema::seal) or [`open`]
    /// (Mnema::open) so repeated seals reuse it instead of re-running the memory-hard Argon2id
    /// KDF on every write. `None` until the store is first sealed or opened.
    ///
    /// [`open`]: Mnema::open
    seal_key: Option<SealingKey>,
}

/// A privacy-oriented census of the store — counts only, never content. Answers the question
/// a user of a local memory actually asks: *"how much Private data am I holding?"* The three
/// episodic tiers partition `total` (`open + redacted + private == total`); `beliefs` and
/// `private_beliefs` count **live** semantic beliefs (superseded history is not counted).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    /// Total episodic memories stored.
    pub total: usize,
    /// Episodic memories at the `Open` tier (shareable content).
    pub open: usize,
    /// Episodic memories at the `Redacted` tier (a sanitized surface may go remote).
    pub redacted: usize,
    /// Episodic memories at the `Private` tier (content never leaves the device).
    pub private: usize,
    /// Live semantic beliefs (contradiction-resolved; superseded history excluded).
    pub beliefs: usize,
    /// Live semantic beliefs held at the `Private` tier.
    pub private_beliefs: usize,
    /// Vectors currently in the exact index (tracks `total` for a well-behaved embedder).
    pub indexed: usize,
}

impl<E: Embedder> Mnema<E> {
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
            seal_key: None,
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
    /// which lifts it against the forgetting curve in [`recall_decayed`](Mnema::recall_decayed).
    /// The redacted surface is left empty, so this is for `Open` or `Private` tiers — a
    /// `Redacted` memory needs [`remember_redacted`](Mnema::remember_redacted) (or
    /// [`remember_with`](Mnema::remember_with)) to carry a surface that can go remote.
    pub fn remember_important(
        &mut self,
        tier: EgressTier,
        importance: f32,
        content: &str,
    ) -> MemoryId {
        self.remember_with(tier, importance, content, "")
    }

    /// Store a `Redacted`-tier episodic memory: `content` is kept locally, but a `Remote`
    /// bundle emits `redacted` in its place (the egress filter's `Redact` decision). This is
    /// the only way to attach a real redacted surface — [`remember`](Mnema::remember) leaves
    /// it empty, so a `Redacted` memory made that way would emit nothing remotely.
    pub fn remember_redacted(&mut self, content: &str, redacted: &str) -> MemoryId {
        self.remember_with(EgressTier::Redacted, 1.0, content, redacted)
    }

    /// Store an episodic memory with full control over all four knobs: egress `tier`,
    /// `importance`, the local `content`, and the `redacted` surface a `Remote` bundle emits
    /// in place of the content for a `Redacted`-tier memory. The surface is ignored for
    /// `Open` (which emits the content) and `Private` (which emits nothing), so it only bites
    /// on the `Redacted` tier. [`remember_important`](Mnema::remember_important) and
    /// [`remember_redacted`](Mnema::remember_redacted) are the common shorthands over this.
    pub fn remember_with(
        &mut self,
        tier: EgressTier,
        importance: f32,
        content: &str,
        redacted: &str,
    ) -> MemoryId {
        // A non-finite importance would poison the forgetting curve: `NaN` makes every
        // `decayed_score`/`is_faded` comparison false — corrupting recall order and leaving the
        // memory *un-prunable* — and `inf` swamps all ranking. Sanitize to neutral salience so a
        // bad input (e.g. a huge JSON number that overflows `f32` to `inf`) can't wreck retrieval.
        let importance = if importance.is_finite() {
            importance
        } else {
            1.0
        };
        let at = self.tick();
        let vector = self.embedder.embed(content);
        let id = self.episodic.append_redacted(
            MemoryKind::Episodic,
            tier,
            at,
            importance,
            content,
            redacted,
        );
        // The index width equals the embedder's dims, so a well-behaved embedder never
        // mismatches; a broken one just leaves this memory unindexed (still stored).
        let _ = self.index.insert(id, vector);
        id
    }

    /// Assert a semantic fact at tier [`EgressTier::Open`], resolving contradictions.
    pub fn remember_fact(&mut self, subject: &str, attribute: &str, value: &str) -> Resolution {
        self.remember_fact_tiered(subject, attribute, value, EgressTier::Open)
    }

    /// Assert a semantic fact with an explicit egress `tier` — a `Private` belief is stored
    /// but never returned to a `Remote` destination (see [`belief_for`](Mnema::belief_for)).
    pub fn remember_fact_tiered(
        &mut self,
        subject: &str,
        attribute: &str,
        value: &str,
        tier: EgressTier,
    ) -> Resolution {
        let at = self.tick();
        self.semantic
            .assert_tiered(subject, attribute, value, at, tier)
    }

    /// The current belief for a `(subject, attribute)` key, if any — the **unfiltered**
    /// read. This can return a `Private` belief; never hand its `value` to a remote model.
    /// Use [`belief_for`](Mnema::belief_for) when assembling a prompt for a destination.
    pub fn belief(&self, subject: &str, attribute: &str) -> Option<&Fact> {
        self.semantic.current(subject, attribute)
    }

    /// The current belief for a key **as visible to `dest`**: a belief the egress filter
    /// would deny (a `Private`, or a `Redacted` bound `Remote`) is withheld — this returns
    /// `None`, exactly as if it did not exist, so a `Remote` model can never read it out.
    pub fn belief_for(&self, subject: &str, attribute: &str, dest: Destination) -> Option<&Fact> {
        self.semantic.current_for(subject, attribute, dest)
    }

    /// Every live belief about `subject` **as visible to `dest`** — answers "what do we know
    /// about X?" across all attributes. Beliefs the egress filter would deny are withheld, so
    /// a `Remote` model never sees a `Private` belief.
    pub fn beliefs(&self, subject: &str, dest: Destination) -> Vec<&Fact> {
        self.semantic.beliefs_for(subject, dest)
    }

    /// Hard-delete every semantic fact matching `predicate` (live or superseded), returning
    /// how many were removed — the right-to-be-forgotten path for beliefs, alongside
    /// [`forget`](Mnema::forget) for episodic memories.
    pub fn forget_facts(&mut self, predicate: impl FnMut(&Fact) -> bool) -> usize {
        self.semantic.forget(predicate)
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
            RetrievalWeights::default(),
        )
    }

    /// Like [`recall`](Mnema::recall), but with explicit per-retriever fusion `weights` — e.g.
    /// [`RetrievalWeights::semantic`] to let a real embedding model's meaning-match outvote a
    /// memory that merely shares a keyword or is more recent. With the default lexical embedder
    /// the dense signal is weak, so weighting it up there hurts more than it helps.
    pub fn recall_weighted(
        &self,
        query: &str,
        dest: Destination,
        per_retriever: usize,
        char_budget: usize,
        weights: RetrievalWeights,
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
            weights,
        )
    }

    /// Like [`recall`](Mnema::recall), but applies the forgetting curve (proposal §3.2):
    /// each hit's score is scaled by its `importance` and a recency weight that halves
    /// every `half_life` ticks, using the facade's own clock as *now*. Recent, important
    /// memories are preferred over stale ones. Note a `half_life` of `0` disables the
    /// *recency decay* (the weight is `1.0`) but **still applies `importance` weighting**,
    /// so it is not identical to [`recall`](Mnema::recall), which applies no scaling at all.
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
            RetrievalWeights::default(),
        )
    }

    /// Build the approximate (IVF) index over the current corpus, enabling
    /// [`recall_fast`](Mnema::recall_fast). `num_anchors` buckets are seeded by
    /// **deterministic k-means** over the corpus embeddings ([`kmeans_anchors`]), so the
    /// anchors sit at the data's real cluster centres and the IVF recovers far more of the
    /// exact top-k than arbitrarily-seeded anchors would; every memory is then bucketed. Call
    /// again to rebuild after the corpus changes — the exact index stays the source of truth
    /// either way.
    ///
    /// [`kmeans_anchors`]: crate::vector::kmeans_anchors
    pub fn build_ann(&mut self, num_anchors: usize) {
        let vectors: Vec<(MemoryId, Vec<f32>)> = self
            .episodic
            .events()
            .iter()
            .map(|e| (e.id, self.embedder.embed(&e.content)))
            .collect();
        let corpus: Vec<Vec<f32>> = vectors.iter().map(|(_, v)| v.clone()).collect();
        let anchors = crate::vector::kmeans_anchors(&corpus, num_anchors, ANN_KMEANS_ITERS);
        let mut ann = IvfIndex::new(self.embedder.dims(), anchors);
        for (id, v) in vectors {
            let _ = ann.insert(id, v);
        }
        self.ann = Some(ann);
    }

    /// Like [`recall`](Mnema::recall), but sources the dense retriever from the
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
            RetrievalWeights::default(),
        )
    }

    /// The `k` most-recent memories by logical time, newest first — the **raw, unfiltered**
    /// view, including `Private` content.
    ///
    /// # Warning
    /// This does **not** apply the egress filter. Never assemble a prompt for a `Remote`
    /// model from its results — that would leak `Private` memories, defeating the ADR-0021
    /// invariant. Use it only for local inspection/debugging, or use
    /// [`recall_recent`](Mnema::recall_recent) to get an egress-safe, budget-bounded bundle.
    pub fn recall_by_recency(&self, k: usize) -> Vec<&Memory> {
        let mut ordered: Vec<&Memory> = self.episodic.events().iter().collect();
        ordered.sort_by_key(|b| std::cmp::Reverse(b.at));
        ordered.truncate(k);
        ordered
    }

    /// The most-recent memories as an **egress-safe** bundle for `dest`, newest first,
    /// bounded to `char_budget` characters. Unlike [`recall_by_recency`](Mnema::recall_by_recency)
    /// this routes through the same choke point as [`recall`](Mnema::recall), so a `Private`
    /// memory's content never reaches a `Remote` bundle.
    pub fn recall_recent(&self, dest: Destination, char_budget: usize) -> Vec<BundleItem> {
        crate::assemble_bundle(self.episodic.events(), dest, char_budget)
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

    /// Strengthen a recalled memory against the forgetting curve by raising its `importance`,
    /// so a memory the agent actually used ranks higher next time and fades slower. Returns
    /// whether a memory with `id` was found. Composes with [`recall_decayed`](Mnema::recall_decayed),
    /// which weights hits by `importance`.
    pub fn reinforce(&mut self, id: MemoryId) -> bool {
        self.episodic.reinforce(id)
    }

    /// Forget every memory that has faded below `threshold` under the forgetting curve — its
    /// `importance × 0.5^(age / half_life)` salience, with the facade's clock as *now*. This is
    /// the counterpart to [`reinforce`](Mnema::reinforce): reinforcement keeps what the agent
    /// uses, pruning sheds what it does not, so a long-lived store stays bounded instead of
    /// growing forever. Hard-deletes through the same choke point as [`forget`](Mnema::forget)
    /// (the vector index is kept in step and ids are never reused), and returns its
    /// [`PurgeReceipt`]. A `half_life` of `0` prunes purely by importance; a `threshold` of
    /// `0.0` prunes nothing.
    pub fn prune_faded(&mut self, half_life: u64, threshold: f32) -> PurgeReceipt {
        let now = self.clock;
        self.forget(|m| is_faded(m.importance, now.saturating_sub(m.at), half_life, threshold))
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

    /// A privacy census of the store — a per-tier count of episodic memories and live beliefs,
    /// exposing *no content*. Lets a user audit how much `Private` data they hold. See
    /// [`MemoryStats`].
    pub fn stats(&self) -> MemoryStats {
        let mut s = MemoryStats {
            total: self.episodic.len(),
            indexed: self.index.len(),
            ..MemoryStats::default()
        };
        for m in self.episodic.events() {
            match m.tier {
                EgressTier::Open => s.open += 1,
                EgressTier::Redacted => s.redacted += 1,
                EgressTier::Private => s.private += 1,
            }
        }
        for f in self.semantic.facts() {
            if f.status == FactStatus::Live {
                s.beliefs += 1;
                if f.tier == EgressTier::Private {
                    s.private_beliefs += 1;
                }
            }
        }
        s
    }

    /// Encrypt the whole memory at rest — the episodic log, the semantic beliefs, and
    /// the logical clock — as one blob (`salt || nonce || AEAD(...)`). The vector index
    /// is *derived* (rebuildable by re-embedding), so it is not stored; [`open`]
    /// reconstructs it. Sealing routes through the same AEAD as the raw store.
    ///
    /// [`open`]: Mnema::open
    /// The passphrase is used to derive the sealing key **once** — on the first seal (or at
    /// [`open`](Mnema::open)); later seals reuse the cached key with a fresh nonce, so a store
    /// that persists on every write pays the memory-hard Argon2id cost once, not per write. (The
    /// cached key stays with this store's original passphrase; to change it, re-seal a freshly
    /// opened store rather than passing a different passphrase here.)
    pub fn seal(&mut self, passphrase: &[u8]) -> Result<Vec<u8>, StoreError> {
        let mut plain = Vec::new();
        // Record the embedder's vector width first, so `open` can refuse a mismatched embedder
        // (ADR-0023) instead of silently rebuilding the index at the wrong width.
        plain.extend_from_slice(&(self.embedder.dims() as u32).to_le_bytes());
        plain.extend_from_slice(&self.clock.to_le_bytes());
        put_bytes(&mut plain, &self.episodic.encode());
        put_bytes(&mut plain, &encode_facts(self.semantic.facts()));
        // Reuse the cached key only if it was derived from THIS passphrase; a new passphrase
        // (e.g. `mnema rekey`) re-derives with a fresh salt.
        let reuse = self
            .seal_key
            .as_ref()
            .is_some_and(|sk| sk.matches(passphrase));
        if !reuse {
            self.seal_key = Some(SealingKey::derive(passphrase)?);
        }
        self.seal_key
            .as_ref()
            .expect("seal key was just derived if absent")
            .seal(&plain)
    }

    /// Recover a whole memory from a [`seal`](Mnema::seal)ed blob with `passphrase` and an
    /// `embedder` whose `dims()` **must** match the one that sealed it — a mismatch is refused
    /// with [`StoreError::EmbedderWidthMismatch`], never silently applied (ADR-0023). The
    /// episodic log, beliefs, and clock are restored verbatim; the vector index is rebuilt by
    /// re-embedding every event, so recall resumes immediately. A wrong key or tampering
    /// yields [`StoreError::Decrypt`].
    pub fn open(blob: &[u8], passphrase: &[u8], embedder: E) -> Result<Self, StoreError> {
        Self::open_inner(blob, passphrase, embedder, true)
    }

    /// Re-embed an existing store under a **new** `embedder` (of any width), preserving every
    /// memory, belief, and the clock — the way to switch embedders (e.g. lexical → semantic)
    /// without losing data. Content is embedder-independent, so only the derived vector index is
    /// rebuilt, at the new width. Seal the returned store to persist it under the new embedder; a
    /// re-seal after this writes the new width, so a later [`open`](Mnema::open) with the new
    /// embedder succeeds. A wrong key or tampering yields [`StoreError::Decrypt`].
    pub fn migrate(blob: &[u8], passphrase: &[u8], embedder: E) -> Result<Self, StoreError> {
        Self::open_inner(blob, passphrase, embedder, false)
    }

    /// Shared decode-and-rebuild behind [`open`](Mnema::open) (`enforce_width = true`: refuse a
    /// mismatched embedder, ADR-0023) and [`migrate`](Mnema::migrate) (`false`: re-embed at the new
    /// width on purpose).
    fn open_inner(
        blob: &[u8],
        passphrase: &[u8],
        embedder: E,
        enforce_width: bool,
    ) -> Result<Self, StoreError> {
        // Derive the key from the blob's salt ONCE and keep it, so subsequent seals of this
        // reopened store skip the Argon2id KDF.
        let seal_key = SealingKey::for_salt(passphrase, SealingKey::salt_of(blob)?)?;
        let plain = seal_key.open(blob)?;
        // On the open path, refuse an embedder whose width differs from the one that sealed the
        // store — opening would rebuild the index at the wrong width and silently corrupt recall
        // (ADR-0023). `migrate` skips this precisely because the width change is intentional.
        let (stored_width, off) = take_u32(&plain, 0)?;
        let embedder_width = embedder.dims() as u32;
        if enforce_width && stored_width != embedder_width {
            return Err(StoreError::EmbedderWidthMismatch {
                stored: stored_width,
                embedder: embedder_width,
            });
        }
        let (clock, off) = take_u64(&plain, off)?;
        let (episodic_bytes, off) = take_bytes(&plain, off)?;
        let (semantic_bytes, _off) = take_bytes(&plain, off)?;

        let episodic = EpisodicLog::decode(episodic_bytes)?;
        let semantic = SemanticStore::from_facts(decode_facts(semantic_bytes)?);

        // The index is derived: rebuild it from the events under `embedder` (same id space; a new
        // width when migrating).
        let mut index = VectorIndex::new(embedder.dims());
        for e in episodic.events() {
            let _ = index.insert(e.id, embedder.embed(&e.content));
        }

        Ok(Self {
            episodic,
            index,
            ann: None, // derived; rebuild explicitly with `build_ann` after opening
            semantic,
            // Working memory is ephemeral — it is not sealed, so open/migrate start fresh.
            working: WorkingMemory::new(WORKING_HORIZON, WORKING_CAPACITY),
            embedder,
            clock,
            seal_key: Some(seal_key),
        })
    }

    /// Write to the ephemeral scratchpad (working memory): a short-lived note stamped
    /// at the current tick. Notes expire after [`WORKING_HORIZON`] ticks and are not
    /// persisted by [`seal`](Mnema::seal).
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

fn tier_tag(tier: EgressTier) -> u8 {
    match tier {
        EgressTier::Open => 0,
        EgressTier::Redacted => 1,
        EgressTier::Private => 2,
    }
}

fn tier_from_tag(tag: u8) -> Result<EgressTier, StoreError> {
    match tag {
        0 => Ok(EgressTier::Open),
        1 => Ok(EgressTier::Redacted),
        2 => Ok(EgressTier::Private),
        _ => Err(StoreError::UnknownTag),
    }
}

/// Serialize semantic facts: per fact `id(8) | at(8) | confidence(4) | status(1) | tier(1)`
/// then the length-prefixed subject, attribute, and value.
fn encode_facts(facts: &[Fact]) -> Vec<u8> {
    let mut buf = Vec::new();
    for f in facts {
        buf.extend_from_slice(&f.id.to_le_bytes());
        buf.extend_from_slice(&f.at.to_le_bytes());
        buf.extend_from_slice(&f.confidence.to_le_bytes());
        buf.push(status_tag(f.status));
        buf.push(tier_tag(f.tier));
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
        let (tier_byte, o) = take_u8(buf, o)?;
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
            tier: tier_from_tag(tier_byte)?,
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
        let mut e = Mnema::new(VowelEmbedder);
        assert_eq!(e.remember(EgressTier::Open, "the cat sat"), 0);
        assert_eq!(e.remember(EgressTier::Open, "the dog ran"), 1);
        assert_eq!(e.len(), 2);
        assert_eq!(e.indexed(), 2);
        assert!(!e.is_empty());
    }

    #[test]
    fn recall_surfaces_the_keyword_match_first() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember(EgressTier::Open, "the cat sat"); // id 0
        e.remember(EgressTier::Open, "the dog ran"); // id 1
        let bundle = e.recall("dog", Destination::Local, 10, 1_000);
        assert_eq!(bundle[0].id, 1);
        assert_eq!(bundle[0].text, "the dog ran");
    }

    #[test]
    fn recall_respects_egress() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember(EgressTier::Private, "the secret dog plan"); // id 0
        e.remember(EgressTier::Open, "a public cat note"); // id 1
        let remote = e.recall("dog", Destination::Remote, 10, 1_000);
        assert!(remote.iter().all(|b| !b.text.contains("secret")));
        let local = e.recall("dog", Destination::Local, 10, 1_000);
        assert!(local.iter().any(|b| b.text == "the secret dog plan"));
    }

    #[test]
    fn remember_with_carries_a_redacted_surface_that_replaces_content_remotely() {
        let mut e = Mnema::new(VowelEmbedder);
        let id = e.remember_with(
            EgressTier::Redacted,
            3.0,
            "full secret dossier",
            "a sealed note",
        );
        // Remote emits the redacted surface in place of the content — never the content itself.
        let remote = e.recall("secret note", Destination::Remote, 10, 1_000);
        assert!(
            remote
                .iter()
                .any(|b| b.id == id && b.text == "a sealed note")
        );
        assert!(remote.iter().all(|b| b.text != "full secret dossier"));
        // Local sees the real content.
        let local = e.recall("secret note", Destination::Local, 10, 1_000);
        assert!(local.iter().any(|b| b.text == "full secret dossier"));
    }

    #[test]
    fn stats_is_a_per_tier_census_of_memories_and_live_beliefs() {
        let mut e = Mnema::new(VowelEmbedder);
        // Distinct per-tier counts (3 / 2 / 1) so no counter can be mistaken for another.
        e.remember(EgressTier::Open, "aaa");
        e.remember(EgressTier::Open, "bbb");
        e.remember(EgressTier::Open, "ccc");
        e.remember_with(EgressTier::Redacted, 1.0, "ddd", "d-surface");
        e.remember_with(EgressTier::Redacted, 1.0, "eee", "e-surface");
        e.remember(EgressTier::Private, "fff");
        // Beliefs: 2 open + 1 private live, so private (1) != non-private (2) — pins the tier test.
        e.remember_fact_tiered("user", "city", "utrecht", EgressTier::Open);
        e.remember_fact_tiered("user", "role", "admin", EgressTier::Open);
        e.remember_fact_tiered("user", "ssn", "sekret", EgressTier::Private);
        // Supersede the city belief: the old fact stays as history but must NOT be counted live.
        e.remember_fact_tiered("user", "city", "amsterdam", EgressTier::Open);

        let s = e.stats();
        assert_eq!(s.total, 6);
        assert_eq!(s.open, 3);
        assert_eq!(s.redacted, 2);
        assert_eq!(s.private, 1);
        // The three episodic tiers partition the total.
        assert_eq!(s.open + s.redacted + s.private, s.total);
        // 3 live beliefs (amsterdam + role + ssn); the superseded utrecht is excluded — pins Live.
        assert_eq!(s.beliefs, 3);
        assert_eq!(s.private_beliefs, 1);
        assert_eq!(s.indexed, 6);
    }

    #[test]
    fn remember_with_neutralizes_a_non_finite_importance() {
        let mut e = Mnema::new(VowelEmbedder);
        let nan = e.remember_with(EgressTier::Open, f32::NAN, "aaa", "");
        let inf = e.remember_with(EgressTier::Open, f32::INFINITY, "bbb", "");
        // With half_life 0, salience == importance. Left as NaN/inf, `is_faded`'s `< threshold`
        // would be false and BOTH would be un-prunable; sanitized to 1.0 they fall below 2.0.
        let receipt = e.prune_faded(0, 2.0);
        assert_eq!(receipt.purged, vec![nan, inf]);
        assert_eq!(receipt.remaining, 0);
    }

    #[test]
    fn forget_removes_from_both_recall_and_index() {
        let mut e = Mnema::new(VowelEmbedder);
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
    fn prune_faded_with_no_decay_sheds_by_importance_and_keeps_the_index_in_step() {
        let mut e = Mnema::new(VowelEmbedder);
        let dull = e.remember_important(EgressTier::Open, 0.5, "idle trivia"); // 0
        let vital = e.remember_important(EgressTier::Open, 5.0, "the dog plan"); // 1
        assert_eq!(e.len(), 2);
        // half_life 0 disables decay, so salience == importance. Cut at 1.0: 0.5 falls, 5.0 stays.
        let receipt = e.prune_faded(0, 1.0);
        assert_eq!(receipt.purged, vec![dull]);
        assert_eq!(receipt.remaining, 1);
        assert_eq!(e.len(), 1);
        assert_eq!(e.indexed(), 1); // the vector index tracked the delete
        // The survivor — the important one — is still recallable.
        let hits = e.recall("dog", Destination::Local, 10, 1_000);
        assert!(hits.iter().any(|b| b.id == vital));
    }

    #[test]
    fn prune_faded_uses_age_when_decay_is_on() {
        let mut e = Mnema::new(VowelEmbedder);
        let old = e.remember(EgressTier::Open, "first note"); // at=1
        let recent = e.remember(EgressTier::Open, "second note"); // at=2; clock is now 2
        // clock=2 → `old` is age 1, `recent` is age 0. With half_life 1 the weights are
        // 0.5 and 1.0; a 0.75 cutoff drops the aged one and keeps the fresh one. This pins
        // the `now - at` direction: computing `at - now` would age neither and prune nothing.
        let receipt = e.prune_faded(1, 0.75);
        assert_eq!(receipt.purged, vec![old]);
        assert_eq!(receipt.remaining, 1);
        let survivors: Vec<MemoryId> = e.recall_by_recency(10).iter().map(|b| b.id).collect();
        assert_eq!(survivors, vec![recent]);
    }

    #[test]
    fn newer_memories_are_more_recent() {
        let mut e = Mnema::new(VowelEmbedder);
        let a = e.remember(EgressTier::Open, "alpha");
        let b = e.remember(EgressTier::Open, "bravo");
        let recent = e.recall_by_recency(2);
        // Pins the monotonic clock: a stalled/rewinding clock would flip this order.
        assert_eq!(recent[0].id, b);
        assert_eq!(recent[1].id, a);
    }

    #[test]
    fn facts_resolve_contradictions_and_belief_reads_the_current_one() {
        let mut e = Mnema::new(VowelEmbedder);
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
        let mut e = Mnema::new(VowelEmbedder);
        e.scratch("first thought");
        e.scratch("second thought");
        let pad: Vec<&str> = e.scratchpad().iter().map(|n| n.content.as_str()).collect();
        assert_eq!(pad, vec!["second thought", "first thought"]);
        // Scratch notes are working memory, not episodic — they don't count as memories.
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn recall_fast_falls_back_to_exact_without_a_built_index() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember(EgressTier::Open, "the cat sat");
        e.remember(EgressTier::Open, "the dog ran");
        // No ANN built → recall_fast must equal the exact recall, bundle for bundle.
        let fast = e.recall_fast("dog", Destination::Local, 10, 1_000, 8);
        let exact = e.recall("dog", Destination::Local, 10, 1_000);
        assert_eq!(fast, exact);
    }

    #[test]
    fn recall_fast_at_full_probe_matches_exact_recall() {
        let mut e = Mnema::new(VowelEmbedder);
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
        let mut e = Mnema::new(VowelEmbedder);
        e.remember(EgressTier::Open, "alpha beta"); // id 0, at 1 (old)
        e.remember(EgressTier::Open, "alpha beta"); // id 1, at 2 (recent)
        // Same content → symmetric base scores; a short half-life lets recency decide.
        let bundle = e.recall_decayed("alpha", Destination::Local, 10, 1_000, 1);
        assert_eq!(bundle[0].id, 1);
    }

    #[test]
    fn remember_important_resists_the_forgetting_curve() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember_important(EgressTier::Open, 10.0, "alpha beta"); // id 0, older but salient
        e.remember(EgressTier::Open, "alpha beta"); // id 1, newer but neutral
        // A long half-life makes decay negligible, so 10× importance keeps the older,
        // salient memory on top despite the newer one.
        let bundle = e.recall_decayed("alpha", Destination::Local, 10, 1_000, 1_000_000);
        assert_eq!(bundle[0].id, 0);
    }

    fn populated() -> Mnema<VowelEmbedder> {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember(EgressTier::Open, "the cat sat"); // id 0
        e.remember(EgressTier::Private, "the secret dog plan"); // id 1
        e.remember_fact("alice", "diet", "vegetarian"); // superseded ↓
        e.remember_fact("alice", "diet", "omnivore"); // live
        e
    }

    #[test]
    fn seal_then_open_restores_memories_beliefs_index_and_clock() {
        let sealed = populated().seal(b"key").unwrap();
        let mut reopened = Mnema::open(&sealed, b"key", VowelEmbedder).unwrap();

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
            Mnema::open(&sealed, b"wrong", VowelEmbedder).err(),
            Some(StoreError::Decrypt)
        );
    }

    #[test]
    fn opening_a_truncated_blob_fails() {
        assert_eq!(
            Mnema::open(&[0u8; 8], b"key", VowelEmbedder).err(),
            Some(StoreError::Truncated)
        );
    }

    #[test]
    fn open_rejects_a_header_length_blob_at_the_sealing_key_boundary() {
        // Short of the version+salt+nonce header (41 bytes): the header split returns Truncated
        // (a `<`→`false` mutant would proceed and mis-report).
        assert_eq!(
            Mnema::open(&[0u8; 20], b"key", VowelEmbedder).err(),
            Some(StoreError::Truncated)
        );
        // Exactly 41 bytes IS a full header, so the length check passes and control reaches the
        // version check; an all-zero blob's version byte (0) is unrecognised → UnknownVersion.
        // Pins the `<` length boundary (a `<=` mutant would mis-report Truncated here).
        assert_eq!(
            Mnema::open(&[0u8; 41], b"key", VowelEmbedder).err(),
            Some(StoreError::UnknownVersion)
        );
    }

    #[test]
    fn open_refuses_an_embedder_of_a_different_width() {
        use crate::embed::HashEmbedder;
        let mut mem = Mnema::new(HashEmbedder::new(64));
        mem.remember(EgressTier::Open, "a stored memory");
        let blob = mem.seal(b"pw").unwrap();
        // the same width reopens fine
        assert!(Mnema::open(&blob, b"pw", HashEmbedder::new(64)).is_ok());
        // a different width is refused, not silently rebuilt at the wrong dimensionality
        assert_eq!(
            Mnema::open(&blob, b"pw", HashEmbedder::new(128)).err(),
            Some(StoreError::EmbedderWidthMismatch {
                stored: 64,
                embedder: 128,
            })
        );
    }

    #[test]
    fn migrate_re_embeds_under_a_new_width_without_data_loss() {
        use crate::embed::HashEmbedder;
        // A store sealed under a width-64 embedder, with memories + a belief.
        let mut mem = Mnema::new(HashEmbedder::new(64));
        mem.remember(EgressTier::Open, "the user prefers TypeScript");
        mem.remember(EgressTier::Private, "api key sk-live-secret");
        mem.remember_fact("user", "diet", "vegetarian");
        let blob = mem.seal(b"pw").unwrap();

        // A width-128 embedder can't OPEN it (refused), but CAN migrate it.
        assert!(Mnema::open(&blob, b"pw", HashEmbedder::new(128)).is_err());
        let mut migrated = Mnema::migrate(&blob, b"pw", HashEmbedder::new(128)).unwrap();

        // Every memory + belief survives; the index is rebuilt at the new width. (Two episodic
        // memories; the fact is a belief, tracked separately.)
        assert_eq!(migrated.len(), 2);
        assert_eq!(migrated.indexed(), 2);
        assert_eq!(
            migrated.belief("user", "diet").map(|f| f.value.as_str()),
            Some("vegetarian")
        );
        let hits = migrated.recall("TypeScript", Destination::Local, 5, 1000);
        assert!(hits.iter().any(|b| b.text.contains("TypeScript")));

        // Re-sealed under the new embedder, it now OPENS at width 128 (migration persisted) and no
        // longer at the old width 64.
        let blob2 = migrated.seal(b"pw").unwrap();
        assert!(Mnema::open(&blob2, b"pw", HashEmbedder::new(128)).is_ok());
        assert!(Mnema::open(&blob2, b"pw", HashEmbedder::new(64)).is_err());
    }

    #[test]
    fn resealing_with_a_new_passphrase_rekeys_the_store() {
        // The cached sealing key must NOT be reused when the passphrase changes (the `mnema
        // rekey` path): re-sealing under a new passphrase re-derives, so the result opens with the
        // NEW passphrase and no longer with the old one. A `==`→`!=` mutant in `matches` would
        // reuse the old key and this would fail.
        let sealed = populated().seal(b"old").unwrap();
        let mut mem = Mnema::open(&sealed, b"old", VowelEmbedder).unwrap();
        let resealed = mem.seal(b"new").unwrap();
        assert!(Mnema::open(&resealed, b"new", VowelEmbedder).is_ok());
        assert_eq!(
            Mnema::open(&resealed, b"old", VowelEmbedder).err(),
            Some(StoreError::Decrypt)
        );
    }

    #[test]
    fn resealing_with_the_same_passphrase_round_trips() {
        // The reuse path (cached key, fresh nonce): sealing twice with the same passphrase both
        // produce openable blobs, and the second (cached) seal is byte-different only in its nonce.
        let mut mem = populated();
        let first = mem.seal(b"pw").unwrap();
        let second = mem.seal(b"pw").unwrap();
        assert!(Mnema::open(&first, b"pw", VowelEmbedder).is_ok());
        assert!(Mnema::open(&second, b"pw", VowelEmbedder).is_ok());
        assert_ne!(
            first, second,
            "a fresh nonce per seal makes the ciphertext differ"
        );
        // The cached key is REUSED (no re-derivation), so its salt — the first 16 bytes — is the
        // same across both seals. If the seal always re-derived, each would draw a fresh salt.
        assert_eq!(first[..16], second[..16], "reused key keeps the same salt");
    }

    #[test]
    fn a_redacted_memory_emits_its_surface_remotely_and_full_content_locally() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember_redacted("card 4111 1111 1111 1111", "card [redacted]");
        // Remote sees the surface, never the full content...
        let remote = e.recall_recent(Destination::Remote, 1_000);
        assert_eq!(remote.len(), 1);
        assert_eq!(remote[0].text, "card [redacted]");
        assert!(!remote[0].text.contains("4111"));
        // ...local sees the full content.
        let local = e.recall_recent(Destination::Local, 1_000);
        assert_eq!(local[0].text, "card 4111 1111 1111 1111");
    }

    #[test]
    fn a_redacted_surface_survives_seal_and_open() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember_redacted("secret detail", "surface");
        let sealed = e.seal(b"key").unwrap();
        let reopened = Mnema::open(&sealed, b"key", VowelEmbedder).unwrap();
        assert_eq!(
            reopened.recall_recent(Destination::Remote, 1_000)[0].text,
            "surface"
        );
    }

    #[test]
    fn a_private_belief_is_withheld_from_a_remote_read() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember_fact_tiered("user", "api_key", "sk-live-123", EgressTier::Private);
        // The unfiltered read sees it; the Remote-facing read must not; Local may.
        assert_eq!(
            e.belief("user", "api_key").map(|f| f.value.as_str()),
            Some("sk-live-123")
        );
        assert!(
            e.belief_for("user", "api_key", Destination::Remote)
                .is_none()
        );
        assert_eq!(
            e.belief_for("user", "api_key", Destination::Local)
                .map(|f| f.value.as_str()),
            Some("sk-live-123")
        );
    }

    #[test]
    fn forget_facts_hard_deletes_matching_beliefs() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember_fact("server", "token", "abc123");
        e.remember_fact("user", "city", "utrecht");
        let purged = e.forget_facts(|f| f.value.contains("abc"));
        assert_eq!(purged, 1);
        assert!(e.belief("server", "token").is_none());
        assert!(e.belief("user", "city").is_some());
    }

    #[test]
    fn recall_recent_is_egress_safe_unlike_recall_by_recency() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember(EgressTier::Private, "the secret plan"); // id 0
        e.remember(EgressTier::Open, "a public note"); // id 1
        // The raw view leaks the private content...
        assert!(
            e.recall_by_recency(2)
                .iter()
                .any(|m| m.content == "the secret plan")
        );
        // ...but the egress-safe recent bundle drops it for a Remote destination.
        let bundle = e.recall_recent(Destination::Remote, 1_000);
        assert!(bundle.iter().all(|b| !b.text.contains("secret")));
        assert!(bundle.iter().any(|b| b.text == "a public note"));
    }

    #[test]
    fn seal_then_open_preserves_a_private_belief_tier() {
        let mut e = Mnema::new(VowelEmbedder);
        e.remember_fact_tiered("user", "api_key", "sk-live-xyz", EgressTier::Private);
        let sealed = e.seal(b"key").unwrap();
        let reopened = Mnema::open(&sealed, b"key", VowelEmbedder).unwrap();
        // The tier byte round-trips: the restored belief is still Private, still withheld
        // from a Remote read (a dropped/mis-decoded tier would leak it).
        assert_eq!(
            reopened.belief("user", "api_key").map(|f| f.tier),
            Some(EgressTier::Private)
        );
        assert!(
            reopened
                .belief_for("user", "api_key", Destination::Remote)
                .is_none()
        );
    }
}
