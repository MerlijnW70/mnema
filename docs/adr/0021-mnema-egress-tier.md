---
id: ADR-0021
title: Per-memory egress tier as the local-first privacy wall
kind: decision
status: accepted
impact: high
domain: security, privacy, memory-layer, local-first
tags: egress, privacy, local-first, exfiltration, prompt-injection, redaction, mnema, memory, threat-model, local-model
relates_to: ADR-0020, BND-tested-not-correct, BND-statistical-quality
supersedes:
superseded_by:
source_parts: 25
decided: 2026-07
summary: Tag every memory with an egress tier; the read-path assembly unconditionally drops a `private`-tier memory bound for a remote destination, making the local-first leak (local store, remote LLM) impossible by construction rather than by discipline. Shipped and ratchet-pinned in src/mnema.rs.
---

# ADR-0021 — Per-memory egress tier

## Context — what problem did we solve?
Local-first storage is *not* privacy on its own. The store lives on-device, but the
moment a retrieved memory is assembled into a prompt for a **cloud** LLM, that content
leaves the machine (proposal §2). A privacy claim that rests on the operator remembering
not to send sensitive memories to a remote model is not a claim — it is a hope. We need
the leak to be **impossible by construction** for the memories the user marks private,
the same way ADR-0007 made the ordering reward-hack *uncompilable* rather than merely
detectable.

## Decision
Every memory carries an **egress tier**: `Open` (may enter any prompt), `Redacted` (a
redacted surface may enter a remote prompt; full form only local), or `Private` (may
enter **only** a local-model prompt, never a remote request). The read path's final stage
is an **egress filter** parameterised by the destination's locality (`Local` / `Remote`):
`assemble_bundle` drops a `Private` memory bound for a `Remote` destination outright, so a
remote `ContextBundle` cannot come to hold private content. The enforcement is
unconditional — there is no runtime flag that disables it.

## Trade-offs — what did we lose to win?
We cap capability for the strictest tier: a `Private` memory is invisible to a cloud
model, so answers that would need it are weaker unless a local model is available. We
accept this — the whole point of the tier is that the user has decided this content is
worth that cost. The trade is **"provable non-egress" over "maximal recall for every
query"**, chosen per-memory by the user, not globally by us.

## Mutation-Coverage Result
**Shipped and pinned** (`src/mnema.rs`). The deterministic invariant —
> *the egress filter never emits a `Private`-tier memory into a bundle whose destination
> is `Remote`* — for all inputs, no exceptions
is enforced in `assemble_bundle` and pinned by `private_content_never_reaches_a_remote_bundle`
(plus the redaction/local-availability tests). This is a hard behavioural contract, unlike
the *quality* of retrieval, which is statistical and belongs on channel B (cf.
BND-statistical-quality). It is exactly the shape the mutation gate pins well: a mutant that
lets a private memory through must be killed by a test — and is, inside the current
**0-survivor** ratchet (90/90 viable killed at time of ratification).

## Consequences — how does this affect future evolution?
The egress filter is a **load-bearing, mutation-pinned invariant** of the read path — on the
same footing as "a resolved contradiction never leaves both versions live" (ADR — the
semantic store). Any future feature that assembles context (summarisation, tool-call
arguments, cross-memory synthesis, the retrieval bundle) MUST route through the egress
filter; a path that bypasses it reverses this ADR. The tier is the user's lever; the
enforcement is not negotiable at runtime.

## Related
- Narrative: `docs/SELF-EVOLUTION.md` Part 25 (the Mnema subsystem)
- ADR-0020 — the dependency budget of the layer this ships in.
- BND-tested-not-correct — the mutation gate proves the filter is *pinned*; a human owns whether the
  tier *policy* is right.
- BND-statistical-quality — why retrieval *quality* is channel-B, but this *contract* is
  ratchet-pinned.
- `docs/proposals/mnema-memory-layer.md` §2, §6c — the local-first tension this closes.
