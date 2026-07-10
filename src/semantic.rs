//! Semantic store with **contradiction-resolving writes** — Phase-1 slice 3
//! (`docs/proposals/engram-memory-layer.md` §3.2, differentiator #1). The place most
//! memory layers are weak: they *append* every fact, so "user is vegetarian" and
//! "user eats meat" both survive and retrieval returns whichever it happens to rank.
//! Engram instead resolves the conflict on write.
//!
//! A fact is a `(subject, attribute) -> value` belief. Asserting a new fact for a key
//! that already has a live belief does one of three things, deterministically:
//!
//! - **agrees** (same value) → reinforce (bump confidence, refresh recency); no dup.
//! - **contradicts and is at least as recent** → supersede: tombstone the old belief
//!   (kept for provenance), install the new one as live.
//! - **contradicts but is stale** (older than the live belief) → keep it as history
//!   only; the fresher belief stands.
//!
//! The load-bearing, noha-pinned invariant — the same shape as ADR-0021's egress rule
//! — is: **at most one *live* fact per `(subject, attribute)`, ever.** A mutant that
//! leaves two live, or lets a stale fact win, must break a test below. Pure safe Rust,
//! zero dependencies (ADR-0007 holds; no `secure` feature needed).
//!
//! Facts carry an [`EgressTier`] just like episodic memories, so the privacy wall covers
//! beliefs too: a `Private` belief is stored but withheld from a remote destination (see
//! [`SemanticStore::current_for`]). Reasserting a belief combines tiers *fail-closed* — the
//! more restrictive one wins — so privacy never silently relaxes.

use crate::{Destination, EgressDecision, EgressTier, egress_decision};

/// Stable handle for a stored fact.
pub type FactId = u64;

/// Whether a fact is the current belief or a superseded/stale record kept for audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactStatus {
    /// The current belief for its `(subject, attribute)` key.
    Live,
    /// A former or stale belief, retained as provenance — never returned by `current`.
    Superseded,
}

/// One `(subject, attribute) -> value` belief, with recency and a confidence that
/// grows each time the belief is independently reasserted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub id: FactId,
    pub subject: String,
    pub attribute: String,
    pub value: String,
    /// Caller-supplied logical timestamp (higher = more recent).
    pub at: u64,
    /// How many times this exact belief has been asserted (starts at 1).
    pub confidence: u32,
    pub status: FactStatus,
    /// The belief's egress class — a `Private` fact is stored but never returned to a
    /// `Remote` destination (see [`SemanticStore::current_for`]).
    pub tier: EgressTier,
}

/// What an [`SemanticStore::assert`] did — precise enough for a caller (or a test) to
/// know whether the belief base actually changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// No prior belief for the key; the fact was recorded as live.
    Inserted,
    /// The live belief already held this value; its confidence and recency grew.
    Reinforced,
    /// The fact contradicted the live belief and was at least as recent; it replaced it.
    Superseded,
    /// The fact contradicted the live belief but was older; kept as history only.
    StaleIgnored,
}

/// A set of semantic beliefs that resolves contradictions on write.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticStore {
    facts: Vec<Fact>,
}

impl SemanticStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every stored fact — live and superseded — in insertion order. The provenance
    /// record, exposed so the whole store can be serialized (e.g. the facade's `seal`).
    pub fn facts(&self) -> &[Fact] {
        &self.facts
    }

    /// Rebuild a store from previously-serialized facts (e.g. the facade's `open`),
    /// preserving their order, statuses, and confidences verbatim.
    pub(crate) fn from_facts(facts: Vec<Fact>) -> Self {
        Self { facts }
    }

    /// Assert `subject.attribute = value` at logical time `at`, tier [`EgressTier::Open`].
    /// See [`assert_tiered`](SemanticStore::assert_tiered) to classify a belief's privacy.
    pub fn assert(
        &mut self,
        subject: impl Into<String>,
        attribute: impl Into<String>,
        value: impl Into<String>,
        at: u64,
    ) -> Resolution {
        self.assert_tiered(subject, attribute, value, at, EgressTier::Open)
    }

    /// Assert `subject.attribute = value` as observed at logical time `at`, with egress
    /// `tier`, resolving against any existing live belief for the key. See the module docs
    /// for the resolution rule. On reinforcement the tier combines *fail-closed*: the more
    /// restrictive of the old and new tiers wins, so a belief's privacy never relaxes by
    /// re-stating it more openly.
    pub fn assert_tiered(
        &mut self,
        subject: impl Into<String>,
        attribute: impl Into<String>,
        value: impl Into<String>,
        at: u64,
        tier: EgressTier,
    ) -> Resolution {
        let subject = subject.into();
        let attribute = attribute.into();
        let value = value.into();

        match self.live_index(&subject, &attribute) {
            None => {
                self.push(subject, attribute, value, at, FactStatus::Live, tier);
                Resolution::Inserted
            }
            Some(i) if self.facts[i].value == value => {
                // Agreement: strengthen the existing belief, do not duplicate it.
                self.facts[i].confidence = self.facts[i].confidence.saturating_add(1);
                self.facts[i].at = self.facts[i].at.max(at);
                self.facts[i].tier = self.facts[i].tier.most_restrictive(tier);
                Resolution::Reinforced
            }
            Some(i) if at >= self.facts[i].at => {
                // Contradiction, at least as recent: the new belief wins.
                self.facts[i].status = FactStatus::Superseded;
                self.push(subject, attribute, value, at, FactStatus::Live, tier);
                Resolution::Superseded
            }
            Some(_) => {
                // Contradiction, but older than the live belief: history only.
                self.push(subject, attribute, value, at, FactStatus::Superseded, tier);
                Resolution::StaleIgnored
            }
        }
    }

    /// The current live belief for a key, if any.
    pub fn current(&self, subject: &str, attribute: &str) -> Option<&Fact> {
        self.live_index(subject, attribute).map(|i| &self.facts[i])
    }

    /// The current live belief for a key **as visible to `dest`** — the egress-filtered
    /// read. A belief whose tier the egress filter would deny (a `Private`, or a `Redacted`
    /// bound `Remote` — facts have no redacted surface) is withheld: this returns `None`,
    /// exactly as if the belief did not exist, so a `Remote` model can never read it out.
    pub fn current_for(&self, subject: &str, attribute: &str, dest: Destination) -> Option<&Fact> {
        self.current(subject, attribute)
            .filter(|f| egress_decision(f.tier, dest) == EgressDecision::Allow)
    }

    /// Hard-delete every fact — live or superseded — matching `predicate`, returning how
    /// many were removed. The right-to-be-forgotten path for beliefs, mirroring the
    /// episodic log's `forget`; a belief thus deleted leaves no live record to supersede.
    pub fn forget(&mut self, mut predicate: impl FnMut(&Fact) -> bool) -> usize {
        let before = self.facts.len();
        self.facts.retain(|f| !predicate(f));
        before - self.facts.len()
    }

    /// Every fact ever recorded for a key — live and superseded — in insertion order.
    /// This is the provenance chain: what was believed, and what replaced it.
    pub fn history(&self, subject: &str, attribute: &str) -> Vec<&Fact> {
        self.facts
            .iter()
            .filter(|f| f.subject == subject && f.attribute == attribute)
            .collect()
    }

    /// All currently-live beliefs, in insertion order.
    pub fn live(&self) -> Vec<&Fact> {
        self.facts
            .iter()
            .filter(|f| f.status == FactStatus::Live)
            .collect()
    }

    /// Index of the live fact for a key, if one exists. The `Live` filter is what
    /// makes "supersede then re-query" return the new belief, not the tombstone.
    fn live_index(&self, subject: &str, attribute: &str) -> Option<usize> {
        self.facts.iter().position(|f| {
            f.status == FactStatus::Live && f.subject == subject && f.attribute == attribute
        })
    }

    fn push(
        &mut self,
        subject: String,
        attribute: String,
        value: String,
        at: u64,
        status: FactStatus,
        tier: EgressTier,
    ) -> FactId {
        let id = self.facts.len() as FactId;
        self.facts.push(Fact {
            id,
            subject,
            attribute,
            value,
            at,
            confidence: 1,
            status,
            tier,
        });
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_count(store: &SemanticStore, subject: &str, attribute: &str) -> usize {
        store
            .history(subject, attribute)
            .iter()
            .filter(|f| f.status == FactStatus::Live)
            .count()
    }

    #[test]
    fn a_first_assertion_is_inserted_as_the_live_belief() {
        let mut s = SemanticStore::new();
        assert_eq!(
            s.assert("alice", "diet", "vegetarian", 1),
            Resolution::Inserted
        );
        assert_eq!(
            s.current("alice", "diet").map(|f| f.value.as_str()),
            Some("vegetarian")
        );
        assert_eq!(s.current("alice", "diet").map(|f| f.confidence), Some(1));
    }

    #[test]
    fn agreement_reinforces_confidence_and_recency_without_duplicating() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegetarian", 1);
        assert_eq!(
            s.assert("alice", "diet", "vegetarian", 5),
            Resolution::Reinforced
        );
        // One belief, not two; confidence grew; recency advanced to the latest sighting.
        assert_eq!(s.history("alice", "diet").len(), 1);
        let cur = s.current("alice", "diet").unwrap();
        assert_eq!(cur.confidence, 2);
        assert_eq!(cur.at, 5);
    }

    #[test]
    fn a_newer_contradiction_supersedes_the_old_belief() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegetarian", 1);
        assert_eq!(
            s.assert("alice", "diet", "omnivore", 2),
            Resolution::Superseded
        );
        // The live belief flipped; both versions survive as provenance; only one live.
        assert_eq!(
            s.current("alice", "diet").map(|f| f.value.as_str()),
            Some("omnivore")
        );
        assert_eq!(s.history("alice", "diet").len(), 2);
        assert_eq!(live_count(&s, "alice", "diet"), 1);
    }

    #[test]
    fn an_equal_age_contradiction_still_supersedes() {
        // Pins the `>=` boundary: at the SAME timestamp the new belief must win.
        // Flipping `>=` to `>` would misroute this to StaleIgnored (old belief stands).
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegetarian", 3);
        assert_eq!(
            s.assert("alice", "diet", "omnivore", 3),
            Resolution::Superseded
        );
        assert_eq!(
            s.current("alice", "diet").map(|f| f.value.as_str()),
            Some("omnivore")
        );
    }

    #[test]
    fn a_stale_contradiction_is_kept_as_history_but_never_wins() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "omnivore", 5);
        // An older, conflicting observation arrives late — it must NOT overwrite.
        assert_eq!(
            s.assert("alice", "diet", "vegetarian", 2),
            Resolution::StaleIgnored
        );
        assert_eq!(
            s.current("alice", "diet").map(|f| f.value.as_str()),
            Some("omnivore")
        );
        // ...but it is retained for audit, as a superseded record.
        assert_eq!(s.history("alice", "diet").len(), 2);
        assert_eq!(live_count(&s, "alice", "diet"), 1);
    }

    #[test]
    fn there_is_never_more_than_one_live_belief_per_key() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegetarian", 1);
        s.assert("alice", "diet", "vegan", 2); // supersede
        s.assert("alice", "diet", "vegan", 3); // reinforce
        s.assert("alice", "diet", "omnivore", 2); // stale, ignored
        s.assert("alice", "diet", "pescatarian", 9); // supersede
        assert_eq!(live_count(&s, "alice", "diet"), 1);
        assert_eq!(
            s.current("alice", "diet").map(|f| f.value.as_str()),
            Some("pescatarian")
        );
    }

    #[test]
    fn distinct_keys_do_not_interfere() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegetarian", 1);
        s.assert("alice", "city", "utrecht", 1); // same subject, different attribute
        s.assert("bob", "diet", "omnivore", 1); // same attribute, different subject
        assert_eq!(
            s.current("alice", "diet").map(|f| f.value.as_str()),
            Some("vegetarian")
        );
        assert_eq!(
            s.current("alice", "city").map(|f| f.value.as_str()),
            Some("utrecht")
        );
        assert_eq!(
            s.current("bob", "diet").map(|f| f.value.as_str()),
            Some("omnivore")
        );
        assert_eq!(s.live().len(), 3);
        // `history` must AND subject with attribute: only alice/diet, not the facts
        // that merely share her subject or the "diet" attribute. Pins `&&` vs `||`.
        assert_eq!(s.history("alice", "diet").len(), 1);
    }

    #[test]
    fn history_preserves_provenance_in_insertion_order() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegetarian", 1);
        s.assert("alice", "diet", "vegan", 2);
        s.assert("alice", "diet", "omnivore", 3);
        let chain: Vec<(&str, FactStatus)> = s
            .history("alice", "diet")
            .iter()
            .map(|f| (f.value.as_str(), f.status))
            .collect();
        assert_eq!(
            chain,
            vec![
                ("vegetarian", FactStatus::Superseded),
                ("vegan", FactStatus::Superseded),
                ("omnivore", FactStatus::Live),
            ]
        );
    }

    #[test]
    fn an_unknown_key_has_no_current_belief() {
        let s = SemanticStore::new();
        assert!(s.current("nobody", "nothing").is_none());
        assert!(s.history("nobody", "nothing").is_empty());
    }

    #[test]
    fn a_default_asserted_fact_is_open_tier() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegan", 1);
        assert_eq!(s.current("alice", "diet").unwrap().tier, EgressTier::Open);
    }

    #[test]
    fn a_private_fact_is_withheld_from_a_remote_read_but_visible_locally() {
        let mut s = SemanticStore::new();
        s.assert_tiered("user", "api_key", "sk-live", 1, EgressTier::Private);
        // Unfiltered read sees it; a Remote read must not; a Local read may.
        assert!(s.current("user", "api_key").is_some());
        assert!(s.current_for("user", "api_key", Destination::Remote).is_none());
        assert_eq!(
            s.current_for("user", "api_key", Destination::Local).map(|f| f.value.as_str()),
            Some("sk-live")
        );
    }

    #[test]
    fn a_redacted_fact_is_withheld_remotely_since_facts_have_no_redacted_surface() {
        let mut s = SemanticStore::new();
        s.assert_tiered("user", "note", "detail", 1, EgressTier::Redacted);
        // Redacted → Remote is a Redact decision, but a fact has no redacted surface, so it
        // is withheld (only an Allow passes current_for).
        assert!(s.current_for("user", "note", Destination::Remote).is_none());
        assert!(s.current_for("user", "note", Destination::Local).is_some());
        // An Open fact is visible to both.
        s.assert_tiered("user", "city", "utrecht", 1, EgressTier::Open);
        assert!(s.current_for("user", "city", Destination::Remote).is_some());
    }

    #[test]
    fn reasserting_a_belief_combines_tiers_fail_closed() {
        let mut s = SemanticStore::new();
        s.assert_tiered("user", "diet", "vegan", 1, EgressTier::Open);
        // Reinforce the SAME value at a more restrictive tier → the tighter tier wins.
        s.assert_tiered("user", "diet", "vegan", 2, EgressTier::Private);
        assert_eq!(s.current("user", "diet").unwrap().tier, EgressTier::Private);
        // Reasserting more openly must NOT relax it back.
        s.assert_tiered("user", "diet", "vegan", 3, EgressTier::Open);
        assert_eq!(s.current("user", "diet").unwrap().tier, EgressTier::Private);
    }

    #[test]
    fn reinforcing_with_an_older_timestamp_does_not_rewind_recency() {
        // Reinforcement takes `at.max(new)`, so re-observing a belief with an OLDER
        // timestamp bumps confidence but must NOT rewind its recency (a `= at` mutant would).
        let mut s = SemanticStore::new();
        s.assert("a", "b", "v", 5);
        assert_eq!(s.assert("a", "b", "v", 2), Resolution::Reinforced);
        let cur = s.current("a", "b").unwrap();
        assert_eq!(cur.at, 5, "recency must not rewind to the older sighting");
        assert_eq!(cur.confidence, 2, "but the belief is still reinforced");
    }

    #[test]
    fn forget_hard_deletes_matching_facts_live_and_superseded() {
        let mut s = SemanticStore::new();
        s.assert("alice", "diet", "vegetarian", 1);
        s.assert("alice", "diet", "omnivore", 2); // supersedes → 2 records for the key
        s.assert("bob", "city", "utrecht", 1);
        let purged = s.forget(|f| f.subject == "alice");
        assert_eq!(purged, 2, "both the live and superseded alice records go");
        assert!(s.current("alice", "diet").is_none());
        assert!(s.current("bob", "city").is_some()); // unrelated belief survives
    }
}
