---
id: ADR-0022
title: Memory-augmented self-evolution — the proposer recalls the loop's own history
kind: decision
status: accepted
impact: high
domain: self-evolution, proposer, memory, feedback
tags: evolve, proposer, memory, engram, recall, history, ledger, feedback, cli, waterline, opt-in, cross-iteration
relates_to: ADR-0006, ADR-0014, ADR-0020
supersedes:
superseded_by:
source_parts:
decided: 2026-07
summary: Give the evolution loop's proposer an opt-in Engram memory of its OWN past iterations — recall the lessons most relevant to this substrate before proposing, remember every verdict after — a queryable, cross-iteration successor to the flat history.json ledger, bridged by a thin `engram` CLI that is I/O glue below noha's behavioral waterline.
---

# ADR-0022 — Memory-augmented self-evolution

## Context — what problem did we solve?
The loop's cross-iteration memory was thin and flat: `history.json` (a proposal-diff
hash + verdict, used only to *verbatim-block* an identical retry — ADR-0006) plus
single-run markers injected into the *next* prompt (the Repair-Request, ADR-0014; the
Plan B analysis). None of it lets the proposer ask "what have I already learned about
*this* substrate?" A lesson from 40 iterations ago — a rejected approach, an accepted
pattern, an uncompilable dead-end — is invisible; the loop can rediscover the same
failure indefinitely. We had just built a memory layer (ADR-0020) whose episodic +
semantic + decay machinery is exactly the missing capability.

## Decision
Give the proposer an **opt-in** (`EVOLVE_MEMORY=1`) Engram memory of the loop's own
history. Before PROPOSE, `evolve.sh` **recalls** the lessons most relevant to the
current substrate/regime and injects them into the prompt ("PRIOR LESSONS — do not
repeat a REJECTED approach; build on an ACCEPTED one"). After the verdict it
**remembers** the outcome (accept summary, or reject reason + detail code). The bridge
is a thin `engram` CLI (`src/bin/engram.rs`): it only parses args, loads/seals the
on-disk store, and prints — **I/O orchestration below noha's behavioral waterline**
(Part 23), exactly like `evolve.sh` itself. All real memory logic is the ratchet-pinned
facade. It degrades to a silent no-op unless enabled, `$ENGRAM_KEY` is set, and the CLI
is built, so every existing loop is byte-for-byte unaffected.

## Trade-offs — what did we lose to win?
A new moving part (a CLI + an encrypted per-campaign store) and a small trust surface:
the CLI glue and the `evolve.sh` wiring are **not** behaviorally probed — they are
orchestration, the same category the project already excludes (Part 23). We accept this
by keeping the CLI genuinely thin (no decisions live in it) and the store git-ignored
and opt-in. The recall *quality* — did it surface the *right* lesson? — is statistical,
a channel-B-shaped concern (BND-statistical-quality), not a ratchet invariant; what IS
pinned is every deterministic contract of the facade and embedder the recall relies on.

## Noha-Fitness-Result — what did the probe say?
The memory layer this rests on is fully pinned: **154/154 viable mutants killed, 0
survivors**, including the new built-in `HashEmbedder`, whose exact FNV mixing is held
by a golden-witness test (ADR-0012's "own your hasher", so no equivalent-mutant escape).
The CLI + `evolve.sh` wiring are glue, verified by demonstration: a real `evolve.sh`
iteration with `EVOLVE_MEMORY=1` printed `MEMORY: injected 2 recalled lesson(s) into the
proposer prompt`; a wrong `$ENGRAM_KEY` cannot open the store; and with the flag off the
run is identical to before (no injection).

## Consequences — how does this affect future evolution?
The loop can now accumulate **durable, queryable** lessons across a long campaign, not
just a verbatim-retry block. Contradiction resolution keeps the belief base current (a
newer "approach X now works" supersedes an older "X failed"); the forgetting curve keeps
recent lessons weighted up. Any future feedback mechanism should prefer *recall* over
adding another flat marker file. The CLI must stay thin — new logic belongs in the
pinned facade, never the glue. The default is off: memory augmentation is a capability
the operator turns on, never a silent change to the certified loop.

## Related
- Narrative: (no `SELF-EVOLUTION.md` Part yet — add on the next write-up)
- ADR-0006 — the anti-reroll `history.json` ledger this augments (verbatim block → recall)
- ADR-0014 — Repair-Request feedback: a per-run signal this generalizes to cross-run memory
- ADR-0020 — the Engram memory layer the proposer now uses
- BND-tested-not-correct — why the recall *quality* (and the glue) sit below the ratchet
