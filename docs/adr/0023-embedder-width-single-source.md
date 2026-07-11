---
id: ADR-0023
title: Pin the default embedder width as one library constant
kind: decision
status: accepted
impact: medium
domain: memory-layer, embeddings, robustness, retrieval
tags: embedder, dims, hashembedder, vector, recall, drift, single-source-of-truth, cli, mcp, waterline, performance-blindness
relates_to: ADR-0020, BND-performance-blindness, BND-statistical-quality
supersedes:
superseded_by:
source_parts: 25
decided: 2026-07
summary: The built-in embedder's dimensionality was a private `const DIMS` duplicated in each binary — the `mnema` CLI on 64, the `mnema-mcp` server on 128 — over one shared store family, so a query vector could be a different width than the stored vectors and silently corrupt recall. Pin it once as `HashEmbedder::DEFAULT_DIMS = 128` in the library and have both binaries reference it, making the agreement structural instead of a convention two crates can drift apart on.
---

# ADR-0023 — One embedder width, pinned in the library

## Context — what problem did we solve?
The built-in `HashEmbedder` (ADR-0012 "own your hasher") takes its width as a runtime
argument, and each binary that constructs one carried its own `const DIMS`. Two binaries
operate the **same store family** — the `mnema` CLI (I/O glue below noha's waterline, ADR-0022)
and the `mnema-mcp` server — and they had drifted apart: the CLI embedded at **64**, the
server at **128**. Sealing/opening a store is dimension-agnostic (it is just crypto over a
blob, ADR-0020), so nothing *fails loudly*. But a query embedded at one width against vectors
stored at another makes cosine similarity meaningless: recall silently degrades to noise. It
is the worst kind of bug — no crash, no error, just quietly wrong retrieval.

Nothing triggered it *yet* (the CLI's evolution-loop caller does not exist in this repo), but
the trap was armed: the day a loop drives `mnema recall` against the server-written store, its
recall breaks with no signal.

## Decision
Pin the default embedder's width **once**, in the library, as the single source of truth:

```rust
impl HashEmbedder {
    pub const DEFAULT_DIMS: usize = 128;
}
```

Both binaries reference `HashEmbedder::DEFAULT_DIMS` instead of a private literal. The value
is **128**, not 64, because the live store was already written at 128 — dropping to 64 would
have orphaned real vectors, whereas the (currently unused) CLI cost nothing to raise. Putting
the constant in the crate both binaries already depend on makes the agreement *structural*:
there is no longer a second number that can be edited in isolation.

## Trade-offs — what did we lose to win?
Very little. A shared constant couples the CLI and server to one library-level choice — but
that coupling is the point: they share data, so they must share the width. The residual gap is
that this pins the *default* width, not a *per-store* width; a caller that deliberately
constructs `HashEmbedder::new(n)` for some other `n` and points it at a `DEFAULT_DIMS` store
still gets silent mismatch (see Consequences).

## Noha-Fitness-Result — what did the probe say?
**Not ratchet-pinnable — and this ADR says so plainly.** This is a textbook
`BND-performance-blindness` case: a behavioral mutation gate cannot see a defect that changes
only *recall quality*, and cross-binary width agreement is exactly that (cf. `BND-statistical-quality`).
The mismatch also lived in the CLI/server glue, which is **below the behavioral waterline**
(ADR-0022) and so is not part of the probed `sources` at all. The `golden_witness_pins_the_exact_mixing`
test pins the FNV mixing at a *fixed* width (`new(8)`); it cannot pin what width two independent
binaries happen to choose. The full suite is green (**83/83** with `--features secure`), but green
here means "the pinned logic still holds," not "the two binaries agree" — the ratchet was never
going to catch this. The guard is therefore **structural** (one constant, impossible to half-edit)
plus review, honestly outside the proof.

## Consequences — how does this affect future evolution?
- Any new process or binary that opens a store in this family MUST embed through
  `HashEmbedder::DEFAULT_DIMS` (or the store's own recorded width) — a fresh private `const`
  reintroduces exactly this drift.
- The deeper, defense-in-depth fix is to **record the embedder width in the store header** and
  refuse to open at a mismatched width — the same "never silently do the wrong thing" discipline
  the per-store keyfile already applies (it refuses to open a store whose key is absent rather
  than clobbering it). That would turn a silent recall-quality failure into a loud open-time
  error, moving the guarantee back below a hard gate. Recorded here as the next step, not taken
  in this change.

## Related
- ADR-0020 — the `secure` feature the CLI and server both build on; the store blob they share.
- ADR-0022 — the CLI as below-waterline glue; why this class of bug is not ratchet-probed.
- ADR-0012 (emerge) — "own your hasher": the witness-pinned `HashEmbedder` this widens.
- BND-performance-blindness / BND-statistical-quality (emerge) — why a behavioral gate is blind
  to a recall-only regression, and why retrieval quality lives on channel B.
