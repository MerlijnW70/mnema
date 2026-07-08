# engram

**A fast, secure, local-first memory layer for LLMs — where every guarantee is proven, not asserted.**

Most memory layers are a vector database with a nice API. `engram` is built around the two things
that actually matter and that most tools get wrong:

1. **Contradiction-resolving writes** — when a new fact conflicts with an old one, it *supersedes*
   with provenance instead of storing both and letting retrieval flip a coin.
2. **Injection-resistant retrieval** — a memory marked `Private` can *never* reach a cloud model;
   stored text is delivered as **data, never as an instruction**.

Both are enforced as hard invariants and pinned by a mutation ratchet (see [Provenance](#provenance)).

```rust
use engram::facade::Engram;
use engram::embed::HashEmbedder;
use engram::{Destination, EgressTier};

let mut mem = Engram::new(HashEmbedder::new(64));

mem.remember(EgressTier::Open,    "user is planning a trip to Japan in spring");
mem.remember(EgressTier::Private, "user's API key is sk-live-abc123");
mem.remember_fact("user", "diet", "vegetarian");
mem.remember_fact("user", "diet", "omnivore");   // supersedes — belief is now "omnivore"

// The Private API key is structurally absent from a bundle bound for a cloud model:
let ctx = mem.recall("what should I know?", Destination::Remote, 5, 2000);

let blob = mem.seal(b"passphrase")?;             // the whole mind, encrypted at rest
let mem  = Engram::open(&blob, b"passphrase", HashEmbedder::new(64))?;   // fully restored
# Ok::<(), engram::store::StoreError>(())
```

## The four memory types

| Type | Module | What it does |
|---|---|---|
| **Episodic** | [`store`](src/store.rs) | timestamped events; **encrypted at rest** (Argon2id + XChaCha20-Poly1305), tamper-evident, hard-deletable |
| **Semantic** | [`semantic`](src/semantic.rs) | facts with **contradiction resolution** — a newer belief supersedes, never accumulates |
| **Working** | [`working`](src/working.rs) | ephemeral scratchpad — TTL horizon + capacity cap; not persisted |
| **Procedural** | *(modeled as semantic facts)* | learned preferences, e.g. "answer in metric" |

## The guarantees

| Invariant | Where |
|---|---|
| A resolved contradiction never leaves two *live* beliefs | `semantic` |
| A `Private` memory's content never reaches a `Remote` bundle | `lib` (egress filter) → one `pack_bundle` choke point |
| Encryption round-trips; a wrong key or tampered byte never yields plaintext | `store` |
| A forgotten memory is gone from state and every re-sealed blob; its id is never reused | `store` |
| At full probe, the approximate index equals the exact oracle | `vector` (`IvfIndex`) |

## Retrieval

Hybrid recall fuses three retrievers with **reciprocal-rank fusion** (`retrieval`):

- **dense** — cosine over embeddings (exact `VectorIndex`, or approximate `IvfIndex`);
- **recency** — newest first;
- **keyword** — lexical overlap.

Then a **forgetting curve** (`recall_decayed`) weights each hit by `importance × 0.5^(age/half_life)`,
and the result is packed through the egress filter under a character budget.

Retrieval is **exact** by default (the correctness oracle). An approximate `IvfIndex` trades recall
for speed and is **opt-in** (`build_ann` + `recall_fast`); at full probe it returns exactly what the
exact index does. Recall-vs-speed is measured, not pinned — run `cargo bench --bench engram`.

## Security

`engram`'s core (retrieval, semantic, working, vector, egress) is **zero-dependency**. Encryption
lives behind an optional **`secure`** feature (three vetted, safe-Rust crates: `chacha20poly1305`,
`argon2`, `getrandom`), which also enables the `Engram` facade, the `engram` CLI, and `seal`/`open`.

- **Egress tiers** — every memory is `Open`, `Redacted`, or `Private`; a `Private` memory bound for a
  `Remote` destination is dropped, unconditionally, at the read path's choke point.
- **Encrypted at rest** — `seal`/`open` derive a key with Argon2id and encrypt with XChaCha20-Poly1305.
- **Hard delete** — `forget` removes matching events from state and every re-sealed blob; ids are
  never reused.
- **Not claimed** — unbreakable, or safe against a compromised OS / root / a malicious local model.
  Local-first raises the bar; it does not make the machine a vault. `#![forbid(unsafe_code)]` crate-wide.

## Quickstart

```bash
# library (core, zero-dependency):
cargo test

# with encryption + the facade:
cargo test --features secure

# the live, self-verifying tour (every guarantee, asserted):
cargo run --example memory_demo --features secure

# the CLI (remember/recall/fact/stats over an encrypted store):
cargo build --features secure --bin engram
export ENGRAM_KEY="your-passphrase"
./target/debug/engram remember mind.engram open "the cat sat on the mat"
./target/debug/engram recall   mind.engram 5 "cat"
```

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — the full architecture, threat model, and phased build plan.
- [`docs/adr/`](docs/adr/) — the architecture decisions: the dependency budget (0020), the egress
  wall (0021), and memory-augmented self-evolution (0022).
- [`docs/NARRATIVE.md`](docs/NARRATIVE.md) — the story of building it under a mutation ratchet.
- [`docs/README.md`](docs/README.md) — how to read the docs (and where cross-references point).

## Provenance

`engram` was extracted from the [**emerge**](https://github.com/MerlijnW70/emerge) research project,
where it was built from commit one under **noha** — a zero-dependency mutation-coverage gate whose
rule is that *a green build proves the changed logic is tested*. Every invariant above is pinned by
that ratchet: at extraction, **155/155 viable mutants killed, 0 survivors, 100%**. Cross-references in
the ADRs to `ADR-0007`, `BND-*`, or `Part N` point back to the emerge repository (see
[`docs/README.md`](docs/README.md)).

## License

Dual-licensed under **MIT OR Apache-2.0**.
