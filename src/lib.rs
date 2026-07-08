#![forbid(unsafe_code)]

//! `engram` — Phase-1 seed of the local-first LLM memory layer
//! (`docs/proposals/engram-memory-layer.md`). The smallest slice that is both
//! *real* and *internal-tool-provable*: the **egress filter** — the load-bearing privacy
//! invariant of ADR-0021 — plus the minimal memory model it guards and
//! the recency assembly it feeds.
//!
//! Deliberately dependency-free and `unsafe`-free (ADR-0007 holds). Encryption at
//! rest and embeddings (ADR-0020, proposed) are **absent on purpose**: this slice
//! does not touch the dependency budget. It proves the *privacy contract* first,
//! because that is the part a deterministic mutation ratchet pins best — and the
//! part the industry gets wrong (a memory sent to a cloud model has left the
//! device; "local storage" is not privacy on its own).

/// Encrypted episodic store (slice 2). Behind the `secure` feature (ADR-0020) so the
/// evolution substrate stays zero-dependency; only the full gate compiles it.
#[cfg(feature = "secure")]
pub mod store;

/// Semantic store with contradiction-resolving writes (slice 3). Pure logic, always
/// compiled — the write-path differentiator (proposal §3.2).
pub mod semantic;

/// Working memory: an ephemeral, self-expiring scratchpad — the fourth memory type
/// (proposal §3.1). Pure logic, always compiled.
pub mod working;

/// Vector retrieval (Phase 2): the pluggable `Embedder` seam + an exact cosine index.
/// Pure logic, always compiled (proposal §3.3).
pub mod vector;

/// A built-in, witness-pinned default `Embedder` (hashed bag-of-words). Zero-dependency
/// so the layer is usable out of the box (ADR-0012 "own your hasher" discipline).
pub mod embed;

/// Hybrid retrieval (Phase 2b): reciprocal-rank fusion of vector + recency + keyword
/// rankings into one `recall`, routed through the egress filter (proposal §3.3).
pub mod retrieval;

/// The `Engram` facade (Phase 2c): one `remember`/`recall`/`forget` API over the whole
/// stack. Behind the `secure` feature — it builds on the encrypted episodic store.
#[cfg(feature = "secure")]
pub mod facade;

/// Where an assembled context bundle is going. `Remote` leaves the device (a cloud
/// LLM); `Local` never does (an on-device model).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Destination {
    Local,
    Remote,
}

/// A memory's egress class — the user's lever (ADR-0021), least to most restrictive.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EgressTier {
    /// May enter any prompt, local or remote.
    Open,
    /// Full form is local-only; a redacted surface may go remote.
    Redacted,
    /// May enter **only** a local-model prompt; never a remote request.
    Private,
}

/// What the egress filter decided for one memory against one destination.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EgressDecision {
    /// Emit the full content.
    Allow,
    /// Emit the redacted surface instead of the content.
    Redact,
    /// Emit nothing — drop the memory from this bundle.
    Deny,
}

/// The kind of store a memory belongs to (proposal §3.1). Not yet load-bearing in
/// this slice — carried so the model is stable as later phases arrive.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryKind {
    Episodic,
    Semantic,
    Procedural,
    Working,
}

/// Stable handle for a stored memory.
pub type MemoryId = u64;

/// One stored memory. Engram never reads the wall clock itself — `at` is a
/// caller-supplied logical timestamp (higher = more recent), which also keeps this
/// module deterministic and testable. (`Eq` is not derived: `importance` is an `f32`.)
#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub id: MemoryId,
    pub kind: MemoryKind,
    pub tier: EgressTier,
    pub at: u64,
    /// A caller-set salience in `[0, ∞)` (default `1.0`). It multiplies a memory's
    /// weight in decayed recall (proposal §3.2) so important memories resist the
    /// forgetting curve; `1.0` is neutral.
    pub importance: f32,
    /// The full content — emitted on `Allow`.
    pub content: String,
    /// The redacted surface — emitted on `Redact` (bound for a remote model).
    pub redacted: String,
}

/// One entry of an assembled bundle: the text actually handed to the model plus the
/// id it came from. Provenance is first-class — memories are *data, not
/// instructions* (proposal §6a).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BundleItem {
    pub id: MemoryId,
    pub text: String,
}

/// The single privacy invariant of ADR-0021, as pure policy: a `Private` memory
/// bound for a `Remote` destination is **denied**, for all inputs, no exceptions.
///
/// This is an enum match with no mutable operators, so the prober finds nothing to
/// flip here directly (a total function with no mutation site) — the invariant earns its proof one
/// layer up, in [`assemble_bundle`], where the deny branch meets real arithmetic.
pub fn egress_decision(tier: EgressTier, dest: Destination) -> EgressDecision {
    match (tier, dest) {
        (EgressTier::Open, _) => EgressDecision::Allow,
        (EgressTier::Redacted, Destination::Local) => EgressDecision::Allow,
        (EgressTier::Redacted, Destination::Remote) => EgressDecision::Redact,
        (EgressTier::Private, Destination::Local) => EgressDecision::Allow,
        (EgressTier::Private, Destination::Remote) => EgressDecision::Deny,
    }
}

/// Assemble a context bundle for `dest`: drop/redact each memory per the egress
/// filter, then keep the most-recent survivors whose combined length fits
/// `char_budget` (proposal §3.3 — recency-ranked, budget-bounded).
///
/// The ADR-0021 guarantee is enforced *here*: no `Private` memory's content ever
/// reaches a `Remote` bundle. Mutating the recency order, the budget accumulation,
/// or the fit test must break a test below — that is what makes the contract
/// *proven*, not merely asserted.
pub fn assemble_bundle(
    memories: &[Memory],
    dest: Destination,
    char_budget: usize,
) -> Vec<BundleItem> {
    // Most-recent first; stable so equal timestamps keep input order.
    let mut ordered: Vec<&Memory> = memories.iter().collect();
    ordered.sort_by(|a, b| b.at.cmp(&a.at));
    pack_bundle(&ordered, dest, char_budget)
}

/// Pack an already-ordered slice of memories into a bundle: apply the egress filter
/// to each in turn, then keep the ones whose running length fits `char_budget`. This
/// is the single choke point every context assembler funnels through — recency
/// assembly ([`assemble_bundle`]) and hybrid retrieval alike — so the ADR-0021
/// guarantee (no `Private` content in a `Remote` bundle) is enforced in exactly one
/// place. The `ordered` order is preserved; only the budget prunes.
pub(crate) fn pack_bundle(
    ordered: &[&Memory],
    dest: Destination,
    char_budget: usize,
) -> Vec<BundleItem> {
    let mut out = Vec::new();
    let mut used = 0usize;
    for &m in ordered {
        let text = match egress_decision(m.tier, dest) {
            EgressDecision::Allow => &m.content,
            EgressDecision::Redact => &m.redacted,
            EgressDecision::Deny => continue,
        };
        let cost = text.chars().count();
        // Skip anything that would overflow the budget; a smaller/lower-ranked item
        // may still fit, so keep scanning rather than breaking.
        if used + cost > char_budget {
            continue;
        }
        used += cost;
        out.push(BundleItem {
            id: m.id,
            text: text.clone(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: MemoryId, tier: EgressTier, at: u64, content: &str) -> Memory {
        Memory {
            id,
            kind: MemoryKind::Semantic,
            tier,
            at,
            importance: 1.0,
            content: content.to_string(),
            redacted: "[redacted]".to_string(),
        }
    }

    #[test]
    fn egress_policy_table_is_exact() {
        use Destination::*;
        use EgressDecision::*;
        use EgressTier::*;
        assert_eq!(egress_decision(Open, Remote), Allow);
        assert_eq!(egress_decision(Open, Local), Allow);
        assert_eq!(egress_decision(Redacted, Local), Allow);
        assert_eq!(egress_decision(Redacted, Remote), Redact);
        assert_eq!(egress_decision(Private, Local), Allow);
        // The invariant, at the policy layer:
        assert_eq!(egress_decision(Private, Remote), Deny);
    }

    #[test]
    fn private_content_never_reaches_a_remote_bundle() {
        let mems = vec![
            mem(1, EgressTier::Private, 10, "SECRET"),
            mem(2, EgressTier::Open, 5, "public"),
        ];
        let bundle = assemble_bundle(&mems, Destination::Remote, 1_000);
        // The private memory is absent entirely; nothing carries its id or content.
        assert!(bundle.iter().all(|b| b.id != 1));
        assert!(bundle.iter().all(|b| b.text != "SECRET"));
        assert!(bundle.iter().all(|b| b.text != "[redacted]")); // Private ≠ Redacted
        assert_eq!(
            bundle,
            vec![BundleItem {
                id: 2,
                text: "public".into()
            }]
        );
    }

    #[test]
    fn private_content_is_available_locally() {
        let mems = vec![mem(1, EgressTier::Private, 10, "SECRET")];
        let bundle = assemble_bundle(&mems, Destination::Local, 1_000);
        assert_eq!(
            bundle,
            vec![BundleItem {
                id: 1,
                text: "SECRET".into()
            }]
        );
    }

    #[test]
    fn redacted_memory_uses_its_redacted_surface_remotely() {
        let mems = vec![mem(1, EgressTier::Redacted, 10, "full detail")];
        let bundle = assemble_bundle(&mems, Destination::Remote, 1_000);
        assert_eq!(
            bundle,
            vec![BundleItem {
                id: 1,
                text: "[redacted]".into()
            }]
        );
        // ...but the full detail is available locally.
        let local = assemble_bundle(&mems, Destination::Local, 1_000);
        assert_eq!(
            local,
            vec![BundleItem {
                id: 1,
                text: "full detail".into()
            }]
        );
    }

    #[test]
    fn bundle_is_most_recent_first() {
        let mems = vec![
            mem(1, EgressTier::Open, 1, "old"),
            mem(2, EgressTier::Open, 9, "new"),
            mem(3, EgressTier::Open, 5, "mid"),
        ];
        let bundle = assemble_bundle(&mems, Destination::Local, 1_000);
        let ids: Vec<MemoryId> = bundle.iter().map(|b| b.id).collect();
        // 9 > 5 > 1 — reversing the recency comparison would reorder this.
        assert_eq!(ids, vec![2, 3, 1]);
    }

    #[test]
    fn budget_admits_only_what_fits_by_recency() {
        // Two 5-char items, budget 5: exactly the most-recent one fits.
        let mems = vec![
            mem(1, EgressTier::Open, 1, "aaaaa"), // older
            mem(2, EgressTier::Open, 9, "bbbbb"), // newer
        ];
        let bundle = assemble_bundle(&mems, Destination::Local, 5);
        // `used + cost > budget`: flipping `>`→`>=` drops the first item (empty);
        // `>`→`<` admits both (budget violated); `+`→`-` underflow-wraps and drops
        // it. Only the true operator yields exactly the newest item.
        assert_eq!(
            bundle,
            vec![BundleItem {
                id: 2,
                text: "bbbbb".into()
            }]
        );
    }

    #[test]
    fn a_zero_budget_admits_nothing() {
        let mems = vec![mem(1, EgressTier::Open, 1, "x")];
        assert!(assemble_bundle(&mems, Destination::Local, 0).is_empty());
    }
}
