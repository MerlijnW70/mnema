---
id: ADR-0020
title: A bounded, vetted, safe-Rust dependency budget for the Mnema memory layer
kind: decision
status: accepted
impact: foundational
domain: dependencies, safety, memory-layer, zero-dependency
tags: dependency, zero-dependency, crypto, rustcrypto, chacha20poly1305, argon2, getrandom, embeddings, mnema, memory, pluggable-embedder, feature-flag, secure
relates_to: ADR-0007, ADR-0021, BND-tested-not-correct
supersedes:
superseded_by:
source_parts: 25
decided: 2026-07
summary: Permit a small, enumerated, safe-Rust crypto set (chacha20poly1305 + argon2 + getrandom) behind an OPTIONAL `secure` feature for the Mnema layer, keeping the evolution substrate zero-dependency by default and probing the gated code via an --all-features gate — a scoped, partial reversal of ADR-0007's zero-dependency wall (the `unsafe` ban is retained in full).
---

# ADR-0020 — A bounded dependency budget for the Mnema memory layer

## Context — what problem did we solve?
The Mnema memory layer (`docs/proposals/mnema-memory-layer.md`) needs capabilities the
zero-dependency crate cannot responsibly self-provide: **encryption at rest** (a cipher +
a memory-hard KDF) and, later, embeddings and an ANN index. "Don't roll your own crypto"
is not style — a hand-rolled cipher is the exact mistake a defensible threat model
(proposal §6) must avoid. ADR-0007's zero-dependency wall was authored for the *evolution
substrate*, where self-containment is the integrity boundary; applied literally to a
crypto boundary it would *force* the mistake it exists to prevent. Per the architectural-
search protocol (step 4), a proposal that re-pays a refused cost is *superseding*, not
extending, and must justify the reversal out loud. This is that justification, ratified.

## Decision
For the **Mnema layer only**, permit a small, explicitly-enumerated, safe-Rust
dependency set behind an **optional `secure` cargo feature** (off by default):
`chacha20poly1305` (XChaCha20-Poly1305 AEAD), `argon2` (Argon2id KDF), `getrandom`
(OS entropy for salts/nonces). The **embedder stays pluggable** (bring-your-own): the
heavy ML dependency is the caller's choice, never a hard dependency of the crate. The
evolution substrate (`cache.rs`, `counter.rs`, `bloom.rs`, `lib.rs`) and the fitness
benches keep building with **zero** dependencies — the wall moves to a *feature
boundary*, it does not fall.

## Trade-offs — what did we lose to win?
We give up the absolute "this crate depends on nothing" claim under `--all-features`, and
take on a supply-chain surface (three pinned, audited crates to review). We accept this
because the alternative — a self-written cipher — is *more* dangerous. The trade is **"a
tiny vetted trusted-computing-base" over "a hand-rolled one,"** and it is scoped: default
builds and the evolution loop are untouched, so ADR-0007's *actual* protection (the loop's
self-contained integrity) is preserved. `unsafe` remains forbidden crate-wide — ADR-0007's
safety half is retained in full; only its zero-dependency half is narrowed to "zero by
default, a named budget under `secure`."

## Noha-Fitness-Result — what did the probe say?
Verified on the first Mnema slice built under this budget (the encrypted episodic store,
`src/mnema/store.rs`). The gate now runs `--all-features` (noha.yaml `test`/`build`/
`guard`, and `check.sh`) so the gated crypto codec is actually probed. The teeth-gap loop
ran as designed: the first full prober reported **8 new survivors** in the manual codec —
of which **2 were genuine equivalent mutants** (`RECORD_HEAD = 8+1+1+8+4` mutated to `20`,
behaviourally indistinguishable because `take_bytes`' own bounds checks made the head-size
unobservable). Rather than baseline the equivalents, the fragile constant was **removed**:
every read now funnels through one checked `take_slice` returning `Truncated` (never an
index panic). Two boundary tests (a `SALT+NONCE`-length blob → `Decrypt`; an over-long
content length → `Truncated`) closed the rest. Re-run: **killed 72/72 viable, 0 survivors,
100%**, `cargo test --all --all-features` = 51 passing, clippy `-D warnings` clean. The
encryption round-trip, wrong-key, and tamper-detection contracts are pinned.

## Consequences — how does this affect future evolution?
The zero-dependency rule is now **per-scope, not per-repo**: the evolution substrate stays
hermetic; the Mnema `secure` feature carries a *named, frozen, safe-Rust* budget. Adding
any crate to that budget is an amendment to *this* ADR — enumerate it, say why a hand-roll
won't do, confirm it is safe Rust. Adding a dependency to the *evolution substrate* still
collides head-on with ADR-0007 and is rejected. Any gated code MUST be probed: the
`--all-features` gate is now part of the boundary, not an optional extra. `unsafe` stays
uncompilable everywhere.

## Related
- Narrative: `docs/SELF-EVOLUTION.md` Part 25 (the Mnema subsystem)
- ADR-0007 — the zero-dependency + forbid-unsafe wall this *partially* reverses (deps
  only; the `unsafe` ban is retained in full).
- BND-tested-not-correct — noha proves the codec is *pinned*; a human ratified the trade.
- `docs/proposals/mnema-memory-layer.md` §7.1 — the tension this resolves.
- ADR-0021 — the companion egress decision; the egress filter code ships in `src/mnema.rs`.
