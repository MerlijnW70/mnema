---
id: ADR-0024
title: Whole-store seal over an append-only log (persistence)
kind: decision
status: accepted
impact: medium
domain: storage, crypto, memory-layer, local-first
tags: persistence, encryption, hard-delete, argon2, append-only, log-structured, engram, storage, sota
relates_to: ADR-0020, ADR-0021
supersedes:
superseded_by:
decided: 2026-07
summary: engram persists the whole store as one re-encrypted blob per write, not an append-only log. With the derived key cached (Argon2id run once per store, not per write) a save is ~1&nbsp;ms at 10k memories and ~11&nbsp;ms at 100k — cheap enough at this layer's scale — and a whole-store rewrite is precisely what makes `forget` a **true immediate physical delete**. An append-only log would defer physical deletion to a later compaction, weakening the hard-delete guarantee, in exchange for a write-latency win that does not matter here. Revisit if save latency at a realistic N becomes a problem, or if the hard-delete guarantee is deliberately relaxed.
---

# ADR-0024 — Whole-store seal over an append-only log

## Context — what problem was raised?

A SOTA benchmark of the local-first agent-memory field (`.sota/`) flags one table-stakes
capability engram does not have: **incremental persistence**. Every direct peer — Perseus
Vault, ai-memory-mcp, mnemo, sqlite-memory, Memoria — writes incrementally to SQLite /
DuckDB / a WAL, so a single memory append touches only that memory. engram instead
re-encodes and re-encrypts the **entire store** on every write (`Engram::seal`), which is
O(N) in the number of memories.

Two facts reframe the question before we "fix" it:

1. **The dominant cost was never the O(N) encode — it was the KDF.** Until recently every
   `seal` re-ran memory-hard Argon2id (~287&nbsp;ms). That is now derived **once** per store
   and cached (a fresh nonce per seal; salt fixed per store). With the key cached, the
   measured whole-store seal is: **N=1k → 0.11&nbsp;ms · 10k → 1.16&nbsp;ms · 50k → 4.94&nbsp;ms ·
   100k → 10.7&nbsp;ms** (`cargo bench --bench persist --features secure`). Linear, but ~0.1&nbsp;µs
   per memory — cheap through and past this layer's realistic scale (a personal/agent memory
   of thousands, not a multi-tenant database).

2. **Whole-store rewrite is *how* engram keeps its hard-delete guarantee.** `forget` is
   documented and pinned as a *true* delete: the forgotten bytes are gone from state and from
   every re-sealed blob, and the id is never reused (ADR-0021 neighbourhood; `store` tests).
   That guarantee exists *because* a save rewrites the whole blob without the forgotten
   records. A log-structured / append-only store cannot delete in place — a "forgotten"
   record's ciphertext lingers on disk until a compaction pass rewrites the log. So going
   incremental does not merely change a data structure; it **weakens a security guarantee**
   unless every `forget` triggers a full compaction (which is exactly the O(N) rewrite we were
   trying to avoid).

## Decision

**Keep whole-store seal.** engram persists by re-encrypting the entire store on each write,
with the Argon2id-derived key cached so the write is just encode + AEAD. `forget` remains an
immediate physical delete by construction. engram will **not** adopt an append-only log for
persistence at this time.

The SOTA "incremental persistence" row is therefore, for engram, a **deliberate design
trade-off — immediate hard-delete over append-only write latency — not an unmet requirement.**

## Consequences

- Writes stay cheap at realistic scale (~1&nbsp;ms at 10k, ~11&nbsp;ms at 100k memories), and the
  `persist` bench tracks this so a regression is visible.
- `forget` stays a true immediate physical delete — no compaction lag, no ciphertext of a
  "deleted" memory left on disk.
- The store remains one portable, encrypted file with no embedded database dependency, which
  keeps the `secure` feature's dependency budget small (ADR-0020).
- engram does not match peers on the literal "incremental write" capability; the scan reflects
  this honestly and points here for the rationale.

## Alternatives considered

- **Append-only sealed segments + compaction-on-forget.** Each `remember` appends one encrypted
  record (O(1)); `forget` compacts (O(N) rewrite) to preserve immediate deletion. This *can*
  keep the hard-delete guarantee, but at the cost of a materially more complex, stateful
  persistence format in correctness-critical crypto code — for a write-latency win (~11&nbsp;ms →
  ~0 at 100k) that is negligible at this layer's scale. **Deferred**, not rejected: this is the
  design to reach for if the revisit criteria below are met.
- **Embedded encrypted KV (`redb` / `sled`) behind `secure`.** Gives incremental insert/delete
  but adds a storage-engine dependency and still defers physical deletion to the engine's own
  compaction. Rejected for now on both the dependency-budget and hard-delete grounds.

## Revisit criteria

Reopen this decision if **either**:

1. Whole-store save latency becomes a real problem at a scale engram actually targets (watch
   the `persist` bench — e.g. if a realistic corpus pushes a save past a human-perceptible
   threshold), **or**
2. The immediate-hard-delete guarantee is deliberately relaxed (at which point append-only
   segments become the natural, and now guarantee-compatible, design).
