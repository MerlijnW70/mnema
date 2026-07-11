# mnema

[![CI](https://github.com/MerlijnW70/mnema/actions/workflows/ci.yml/badge.svg)](https://github.com/MerlijnW70/mnema/actions)

**A fast, secure, local-first memory layer for LLMs — where every guarantee is proven, not asserted.**

Most memory layers are a vector database with a nice API. `mnema` is built around the two things
that actually matter and that most tools get wrong:

1. **Contradiction-resolving writes** — when a new fact conflicts with an old one, it *supersedes*
   with provenance instead of storing both and letting retrieval flip a coin.
2. **Injection-resistant retrieval** — a memory marked `Private` can *never* reach a cloud model;
   stored text is delivered as **data, never as an instruction**.

Both are enforced as hard invariants and pinned by a mutation ratchet (see [Provenance](#provenance)).

```rust
use mnema::facade::Mnema;
use mnema::embed::HashEmbedder;
use mnema::{Destination, EgressTier};

let mut mem = Mnema::new(HashEmbedder::new(64));

mem.remember(EgressTier::Open,    "user is planning a trip to Japan in spring");
mem.remember(EgressTier::Private, "user's API key is sk-live-abc123");
mem.remember_fact("user", "diet", "vegetarian");
mem.remember_fact("user", "diet", "omnivore");   // supersedes — belief is now "omnivore"

// The Private API key is structurally absent from a bundle bound for a cloud model:
let ctx = mem.recall("what should I know?", Destination::Remote, 5, 2000);

let blob = mem.seal(b"passphrase")?;             // the whole mind, encrypted at rest
let mem  = Mnema::open(&blob, b"passphrase", HashEmbedder::new(64))?;   // fully restored
# Ok::<(), mnema::store::StoreError>(())
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
exact index does. Recall-vs-speed is measured, not pinned — run `cargo bench --bench mnema`.

## Security

`mnema`'s core (retrieval, semantic, working, vector, egress) is **zero-dependency**. Encryption
lives behind an optional **`secure`** feature (three vetted, safe-Rust crates: `chacha20poly1305`,
`argon2`, `getrandom`), which also enables the `Mnema` facade, the `mnema` CLI, and `seal`/`open`.

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
cargo build --features secure --bin mnema
export MNEMA_KEY="your-passphrase"
./target/debug/mnema remember mind.mnema open "the cat sat on the mat"
./target/debug/mnema recall   mind.mnema 5 "cat"
```

## Use it as an MCP server

Give any MCP client (Claude Code, Cursor, Claude Desktop) private, local, encrypted memory.
Because `recall` is egress-filtered, a `Private` memory **never reaches the model**:

```bash
cargo build --release --features mcp --bin mnema-mcp
```

Point your MCP client at the binary, with the store path and passphrase in the environment:

```jsonc
{
  "mcpServers": {
    "mnema": {
      "command": "/path/to/mnema-mcp",
      "env": { "MNEMA_PATH": "~/mnema.store", "MNEMA_KEY": "your-passphrase" }
    }
  }
}
```

Tools: **remember** (with an `open` / `redacted` / `private` tier), **recall**,
**remember_fact**, **forget**, **stats**. The server is thin glue over the library — every
guarantee above still holds, and the memory logic stays mutation-pinned.

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — the full architecture, threat model, and phased build plan.
- [`docs/adr/`](docs/adr/) — the architecture decisions: the dependency budget (0020), the egress
  wall (0021), memory-augmented self-evolution (0022), and whole-store-seal persistence (0024).
- [`docs/NARRATIVE.md`](docs/NARRATIVE.md) — the story of building it under a mutation ratchet.
- [`docs/README.md`](docs/README.md) — how to read the docs (and where cross-references point).

## Provenance

`mnema` was extracted from the [**emerge**](https://github.com/MerlijnW70/private) research project,
where it was built from commit one under **internal-tool** — a zero-dependency mutation-coverage gate whose
rule is that *a green build proves the changed logic is tested*. Every invariant above is pinned by
that ratchet: at extraction, **155/155 viable mutants killed, 0 survivors, 100%**. Cross-references in
the ADRs to `ADR-0007`, `BND-*`, or `Part N` point back to the emerge repository (see
[`docs/README.md`](docs/README.md)).

## Validated in a live agentic loop

Beyond its unit tests, `mnema` has been driven by a **real self-improving agent** (emerge's
evolution loop, see [`docs/adr/0022`](docs/adr/0022-memory-augmented-self-evolution.md)): the loop
stored each iteration's verdict as a memory and recalled the relevant lessons before the next
proposal. In live runs the proposer **demonstrably used the recall** — it cited a recorded rejection
by its detail code (`channelB-regress:scan`) and redesigned its change specifically so it would *not*
repeat the remembered failure. Contradiction-resolving writes kept the belief base current across
iterations; the egress filter and encrypted store held throughout. The memory layer works not just in
a test harness, but as the actual memory of an agent doing real work.

## License

Dual-licensed under **MIT OR Apache-2.0**.
