## Part 25 — Engram: building a memory layer under the same discipline (July 2026)

Parts 6–24 evolved a *substrate* (a cache, a counter, a Bloom filter) to prove the loop. Part 25 asks a
different question: does "green means proven" hold up when you build a **real product** — a fast,
secure, local-first memory layer for LLMs — from commit one? The proposal lives in
`docs/proposals/engram-memory-layer.md`; the code is the `engram` module. The bet was that the two
things every memory layer gets wrong — **letting contradictory facts both survive**, and **letting a
stored "memory" become an instruction** (prompt injection via retrieval) — are exactly the kind of
*deterministic invariant* noha's ratchet pins best, and so could be made *provable* rather than merely
asserted. They were.

**Two decisions had to be paid for out loud.** [`ADR-0020`](evolution/adr/0020-engram-dependency-budget.md)
partially reverses the zero-dependency wall ([`ADR-0007`](evolution/adr/0007-forbid-unsafe-concurrency-wall.md)):
a cipher cannot be responsibly hand-rolled, so a small, enumerated, safe-Rust crypto set (`chacha20poly1305`,
`argon2`, `getrandom`) is allowed — but *only* behind an optional `secure` feature, so the evolution
substrate and benches stay zero-dependency by default, and the full gate probes the gated code via
`--all-features`. The `unsafe` ban is retained in full. [`ADR-0021`](evolution/adr/0021-engram-egress-tier.md)
makes the local-first leak (local store, remote LLM) *impossible by construction*: every memory carries an
egress tier, and the read path unconditionally drops a `Private` memory bound for a `Remote` destination —
the same "make the reward-hack uncompilable, don't just detect it" move as the concurrency wall, one layer up.

**The invariants that are now ratchet-pinned**, each the shape noha kills well:

- *A resolved contradiction never leaves two live beliefs.* The semantic store supersedes on a newer-or-equal
  conflict, keeps the old as provenance, and ignores a stale one — pinned to the exact `>=` boundary.
- *A `Private` memory's content never reaches a `Remote` bundle* — enforced at one `pack_bundle` choke point
  that both recency assembly and hybrid retrieval funnel through, so retrieval can't become a privacy backdoor.
- *Encryption round-trips; a wrong key or a tampered byte never yields plaintext* (Argon2id + XChaCha20-Poly1305).
- *A forgotten memory is gone from state and from every re-sealed blob, and its id is never reused.*

**The teeth-gap loop did its job three times.** The manual codec's `RECORD_HEAD = 8+1+1+8+4` constant bred
two genuine *equivalent* mutants (the head size was unobservable behind `take_bytes`' own bounds checks);
rather than baseline them, the fragile constant was **removed** and every read routed through one checked
`take_slice` — the AGENTS.md-step-3 "restructure so the equivalent disappears" move (first seen in Part 6).
A `&&`→`||` in the fact-history filter and an `==`→`!=` in the fused-id→memory lookup were invisible to
set-membership assertions and only died once a test pinned *rank order* — a reminder that a mutation survives
until a test observes the exact thing it changes.

**And the boundary from Part 23 reappeared, by design.** Retrieval is built on *exact* nearest-neighbour, not
HNSW. An ANN index is faster but only *approximately* correct — its win is latency, which is **invisible to
noha's behavioral ratchet** (BND-performance-blindness, the same wall as the memory-ordering one). So exact
brute-force is the *correctness oracle* the ratchet pins, and its O(N) scan is measured on **channel B**
(`benches/engram.rs`, `scripts/fitness-engram.sh`), with a planted nearest neighbour as the behavioral pin.
That bench is exactly where an approximate index has to prove itself — and one now does: an inverted-file
`IvfIndex` that scans only the query's nearest buckets. It is **ratchet-pinned to equal the exact oracle at
full probe** (a mutant that drops a bucket or mis-assigns a vector breaks that equality), while its recall is
a channel-B quality objective, not an invariant — exactly the Bloom-filter split of Parts 18–20. The bench
measures the trade: on a 5k×64 workload it runs ~2.6× faster than the exact scan at a reported recall. The
memory layer, like noha itself, knows precisely what its gate cannot see.

## Open questions (still unanswered)

1. Does the ratchet resist a *determined* optimizer that inflates mutation score with tautological tests
   at scale (beyond the blind spots we've already closed)? *(Part 7's ten-iteration stress test:
   no reward-hacking and no false accepts observed. The one norms-not-mechanism gap it identified —
   the re-roll temptation — is now mechanized (Part 9). Hundreds of iterations remain untested.)*
2. How are **equivalent mutants** (undecidable in general) adjudicated at scale without a human per
   case? *(First empirical data point in Part 6: on this repo's first real iteration, restructuring
   the code so the equivalent branch disappeared beat recording an exception — but that may not
   generalize to shapes with no branch-free rewrite.)*
3. Do co-evolving/adversarial evaluators durably beat reward hacking, or just relocate the Goodhart
   problem into a new arms race?
4. What are the real failure/containment boundaries — distribution shift, local-optima entrapment,
   compute-vs-gain scaling — beyond curated benchmarks?

## Primary sources

- Darwin Gödel Machine — https://arxiv.org/abs/2505.22954 · https://sakana.ai/dgm/
- AlphaEvolve — https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- ADAS / Meta Agent Search — https://arxiv.org/abs/2408.08435
- Voyager — https://arxiv.org/abs/2305.16291
- Reward-model overoptimization (Goodhart scaling laws) — https://arxiv.org/abs/2210.10760
- Reward-hacking survey (2026) — https://arxiv.org/html/2604.13602v1
- JudgeDeceiver (gaming LLM judges) — https://arxiv.org/abs/2403.17710
- RQGM (co-evolving evaluator) — https://arxiv.org/pdf/2606.26294
- Mutation vs real faults — https://homes.cs.washington.edu/~mernst/pubs/mutation-effectiveness-fse2014.pdf
- Mutation-vs-faults counterpoint — https://dl.acm.org/doi/10.1145/3180155.3180183

---

*Generated by a 104-agent deep-research harness (adversarially verified), then grounded against this
repo's own red-team results. See `AGENTS.md` for the rules every contributor — human or agent — follows.*
