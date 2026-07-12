# Proposal: **Mnema** — a fast, secure, local-first memory layer for LLMs

> **Status: HISTORICAL PROPOSAL — superseded by the implementation.** This was the
> "design doc first" RFC, written *before* the code, and kept for provenance. It is **not**
> the current source of truth, and several parts were deliberately built differently or not
> at all. For what Mnema actually is, read the [README](../README.md), the [ADRs](adr/), and
> the source. Known, intentional divergences from this proposal:
>
> - **Vector index:** HNSW → an exact `VectorIndex` (the correctness oracle) plus an opt-in
>   approximate `IvfIndex`; ANN recall is a channel-B (measured, not ratchet-pinned) concern.
> - **Keyword retrieval:** full BM25 → a simpler distinct-term-overlap `keyword_rank`.
> - **Context assembly:** MMR / knapsack diversity → recency + character-budget packing.
> - **Persistence:** an append-only log → a **whole-store seal** (ADR-0024), chosen for true
>   immediate hard-delete. Every blob now carries a format-version byte and the embedder width.
> - **Not built:** the entity/relation **graph** index, **bloom/vector write-time dedup**, and
>   a write-time injection classifier. Retrieval delivers memories as *data*, never instructions.
>
> Codename **Mnema** (a stored memory trace) — a memory engine built next to `emerge` under
> its correctness discipline (noha: a green build proves the changed logic is *tested*).

---

## 1. The claim, made honest

The ask was: *"extremely fast, 100% secure, world-class memory, all features an LLM
needs — unmatched."* Two of those are achievable and one is not, so we restate the
goal as something we can actually defend:

| Asked for | What we will actually claim | Why the restatement |
| --- | --- | --- |
| Extremely fast | **Sub-millisecond recall on 100k+ memories, on-device, single-digit-ns hot ops** | Real, measurable, and our Rust substrate already lives here. |
| 100% secure | **Encrypted at rest · injection-resistant retrieval · explicit egress control · auditable · true hard-delete** | "100% secure" is unfalsifiable marketing. A concrete threat model is defensible. |
| World-class / unmatched | **Provably-tested memory logic (noha) + zero-`unsafe` Rust + local-first** | The moat is *correctness we can prove*, not feature count. No competitor has this. |

**The one differentiator that matters:** plenty of agent-memory tools already exist and all
have "features." "Unmatched" cannot come from a longer feature list. It comes from the two
things those tools do *badly* and we can do *provably*:

1. **Contradiction-resolving writes** — when a new fact conflicts with an old one,
   supersede it with provenance; don't accumulate both. Most tools just append.
2. **Injection-resistant retrieval** — stored text is *data*, never *instructions*.
   Memory poisoning is the marquee AI-memory attack and the industry is weak on it.

Both are decision-heavy code paths — exactly what noha's mutation ratchet pins. That
is the whole thesis: *our memory engine's judgment is tested, not just its plumbing.*

---

## 2. Local-first: what it does and does not buy us

Local-first (chosen deployment) removes the two hardest security problems outright —
there is **no network trust boundary** and **no multi-tenant isolation**. That is a
large, real win.

But it introduces one tension we must name honestly and design around:

> **The store is on-device. The *use* may not be.** If you feed retrieved memories to
> a **cloud** LLM (Claude, GPT, …), that context leaves the machine. "Local storage"
> is *not* "private" on its own.

So egress is a first-class, designed concern (§6), not an afterthought. A memory can
be tagged with an egress tier; sensitive tiers are usable only by a **local** model or
are redacted before they can enter a remote prompt. We make the leak *impossible by
construction* for the tiers the user marks private, rather than promising discipline.

---

## 3. Architecture

```
                          ┌─────────────────────────────────────────┐
   remember(text) ─────▶  │  WRITE PATH                              │
                          │  extract → embed → dedup(bloom+vec)      │
                          │  → contradiction-resolve → decay-sched    │
                          └───────────────┬──────────────────────────┘
                                          ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  STORAGE CORE  (encrypted at rest, XChaCha20-Poly1305)            │
   │  ┌────────────────────┐   ┌──────────────────────────────────┐    │
   │  │ Append-only event  │   │ Derived indexes (rebuildable)     │    │
   │  │ log = source of     │──▶│  • HNSW vector index              │    │
   │  │ truth + audit +     │   │  • BM25 keyword index             │    │
   │  │ time-travel + purge │   │  • entity/relation graph          │    │
   │  └────────────────────┘   │  • hot-memory LRU  (cache.rs)     │    │
   │                            │  • dup pre-filter  (bloom.rs)    │    │
   │                            └──────────────────────────────────┘    │
   └───────────────────────────────────┬──────────────────────────────┘
                                        ▼
                          ┌─────────────────────────────────────────┐
   recall(q, budget) ◀──  │  READ PATH                               │
                          │  hybrid retrieve (vec ⊕ bm25 ⊕ graph)    │
                          │  → RRF fuse → rerank(relev×recency×imp)   │
                          │  → MMR context assembly under token budget│
                          │  → egress filter → provenance-tagged      │
                          └─────────────────────────────────────────┘
```

### 3.1 Memory model (four stores, human-memory-inspired)

| Store | Holds | Lifetime |
| --- | --- | --- |
| **Episodic** | timestamped events/turns — the raw log | permanent unless purged |
| **Semantic** | extracted facts, entities, relations (a small graph) | superseded on contradiction |
| **Procedural** | learned preferences / how-to ("always answer in metric") | superseded on change |
| **Working** | ephemeral scratchpad for the current session | TTL / never persisted |

The **episodic log is the single source of truth**; every other index is a derived,
rebuildable projection. This gives us audit, time-travel, cheap schema evolution, and
— critically — a real basis for hard-delete via log compaction (§6d).

### 3.2 Write path — *differentiator #1*

`remember(input)`:
1. **Extract** candidate memories from raw input (LLM-assisted or rule-based; pluggable).
2. **Embed** each candidate (pluggable embedder — see §7).
3. **Dedup** — `bloom.rs` as a cheap pre-filter to reject exact/near-exact repeats
   before the expensive vector near-duplicate check.
4. **Contradiction-resolve** — detect when a candidate conflicts with an existing
   memory (same subject, incompatible predicate). On conflict: write a new version,
   **tombstone the old with provenance**, keep confidence + timestamp. *Do not append
   both.* This is the function that most determines quality and is the first noha
   target.
5. **Schedule decay** — assign an importance and a forgetting curve; low-value
   memories fade unless reinforced.

### 3.3 Read path — *differentiator #2*

`recall(query, token_budget)`:
1. **Hybrid retrieve** — HNSW vector + BM25 keyword + one graph hop, in parallel.
2. **Fuse** — reciprocal rank fusion (hybrid beats pure-vector, reliably).
3. **Rerank** — `score = relevance × recency_decay × importance`.
4. **Assemble** — a knapsack/MMR selection that maximizes marginal relevance under the
   token budget (avoid five memories that all say the same thing).
5. **Egress filter** (§6) + attach provenance, then hand back a `ContextBundle` whose
   memories are marked as *data, not instructions*.

---

## 4. Speed plan (reuses what `emerge` already proved)

| Technique | Source in this repo |
| --- | --- |
| Bloom pre-filter before vector dedup | `src/bloom.rs` (already ~3.3× faster hashing via `FastHasher`) |
| Hot-memory LRU cache (~7 ns/op) | `src/cache.rs` (SoA links + hole-free slab) |
| HNSW ANN for vector recall | new; safe-Rust implementation |
| Product-quantized vectors | new; shrinks the index, keeps recall high |
| Concurrency | `src/counter.rs` sharding pattern; **safe Rust only** (ADR-0007 wall holds) |

Target: recall p50 < 1 ms over 100k memories on a laptop; write p50 < a few ms
(dominated by embedding, which is pluggable/batchable).

---

## 5. API surface

```rust
mnema.remember(input)                 -> Vec<MemoryId>   // extract+dedup+resolve internally
mnema.recall(query, budget)           -> ContextBundle   // hybrid + assembled, egress-filtered
mnema.forget(filter)                  -> PurgeReceipt     // TRUE hard-delete (log compaction)
mnema.revise(...)                     // handled inside remember(); no manual "update both"
mnema.audit(memory_id)                -> History          // provenance + version chain
```

Plus an **MCP server** (`mnema-mcp`) so any agent uses it tool-to-tool — the exact
pattern this repo already ships for `noha-mcp`. That is how "any LLM" gets the memory,
without an SDK per language.

---

## 6. Threat model (what "secure" concretely means)

We defend a **single-owner on-device** deployment. Named adversaries and defenses:

**a. Memory poisoning / prompt injection (the important one).**
Attack: text stored as a "memory" is later retrieved and silently interpreted as
*instructions* to the model ("ignore previous instructions, exfiltrate…").
Defense: structural — retrieved memories are delivered in a **data channel** with a
fixed "these are quotes, never commands" contract, carry provenance, and (optional)
pass an injection classifier on write. The model is never handed a memory as a bare
instruction. This is designed-in, not bolted-on.

**b. At-rest theft (stolen device / imaged disk).**
Defense: whole-store encryption (XChaCha20-Poly1305), key derived from an OS-keychain
secret or passphrase (Argon2id). Indexes are encrypted too — a vector index leaks
semantic content otherwise.

**c. Exfiltration through the LLM itself (the local-first tension, §2).**
Defense: per-memory **egress tier**. `private` tier memories are usable only by a
local model or are redacted before entering any remote prompt. Enforced at the read
path's egress filter, so a private memory *cannot* be assembled into a cloud request.

**d. Right-to-be-forgotten / PII.**
Defense: `forget()` performs real **log compaction** — the bytes are gone, not just
tombstoned — and rebuilds the derived indexes. Auditable purge receipt.

**What we explicitly do NOT claim:** unbreakable, "100% secure," or safe against a
compromised OS / root attacker / a malicious local model. Local-first raises the bar a
lot; it does not make the machine a vault.

---

## 7. Honest tensions with `emerge`'s ethos (decide these first)

Two of this repo's rules are load-bearing and this proposal strains both. These are
the real design decisions, and they should become ADRs before any code:

1. **Zero-dependency vs. crypto + embeddings + ANN.** We cannot hand-roll a cipher
   responsibly. Recommendation: allow a **bounded, vetted, pure-safe-Rust** dependency
   set for the Mnema crate only (RustCrypto for the cipher/KDF — all `#![forbid(unsafe)]`-
   compatible), behind a feature flag, and keep the *embedder pluggable* (bring-your-own,
   e.g. a local gguf/ONNX model) so the heavy ML dependency is the caller's choice, not
   ours. This preserves "green means proven" for *our* logic while not pretending we
   reimplement AES from scratch.

2. **`#![forbid(unsafe_code)]` vs. SIMD vector math.** Distance kernels often want
   `unsafe` SIMD. Recommendation: keep the wall (ADR-0007). Safe Rust autovectorizes
   acceptably; if we need more, use a safe SIMD abstraction, never raw intrinsics. The
   wall is the moat — do not trade it for a benchmark.

3. **noha coverage of *probabilistic* judgment.** Just like `bloom.rs`'s false-positive
   rate (Part 18–20), the *quality* of contradiction-resolution and ranking is
   statistical, not a deterministic contract a mutant can violate. So: pin the
   deterministic invariants with noha (a resolved contradiction never leaves both
   versions live; egress filter never emits a `private` memory to a remote bundle;
   `forget` leaves zero recoverable bytes) **and** guard the fuzzy quality with a
   channel-B fitness benchmark (recall@k, contradiction-accuracy on a labeled set).

---

## 8. Phased build plan

| Phase | Deliverable | noha pins |
| --- | --- | --- |
| **0** | This doc → 2–3 ADRs (dep policy, egress model, unsafe wall reaffirmed) | — |
| **1** | Storage core: encrypted append-log + slab index + episodic store + recall-by-recency | log integrity, encryption round-trip, purge |
| **2** | Embeddings (pluggable) + HNSW + BM25 + hybrid retrieve + MMR assembly | fusion determinism, budget never exceeded |
| **3** | Write intelligence: bloom+vec dedup, **contradiction resolution**, decay | "never both versions live"; dedup threshold |
| **4** | Security hardening: injection-resistant retrieval contract, egress tiers, hard-delete/compaction | egress filter never leaks `private`; zero-byte purge |
| **5** | `mnema-mcp` server + a reference local-model integration | protocol contract |

Each phase ends only when `scripts/check.sh` is green (fmt · clippy · test · full
ratchet · manifest) — same bar as the rest of the repo.

---

## 9. What I need from you to start Phase 0

1. **Ratify the dependency policy** (§7.1) — bounded vetted safe-Rust deps + pluggable
   embedder? Or hold a stricter line?
2. **Confirm the egress model** (§2, §6c) — is per-memory egress-tier + local-model
   enforcement the right shape, or do you want local-model-*only* for everything
   (simpler, but caps capability)?
3. **Pick the first vertical slice** — I'd build Phase 1 (encrypted episodic store +
   recall-by-recency) as the smallest thing that is *real and provable*, then layer
   retrieval on top.

If those three land, I'll turn §7 into ADRs and scaffold the Phase-1 crate.
