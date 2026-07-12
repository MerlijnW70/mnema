# mnema

[![CI](https://github.com/MerlijnW70/mnema/actions/workflows/ci.yml/badge.svg)](https://github.com/MerlijnW70/mnema/actions)
![wasm](https://img.shields.io/badge/wasm-ready-8A2BE2.svg)
![forbid(unsafe)](https://img.shields.io/badge/unsafe-forbidden-success.svg)

**A fast, secure, local-first memory layer for LLMs — where every guarantee is proven, not asserted.**

The zero-dependency core compiles to **WebAssembly**, so mnema can run entirely in a browser
tab — memory that never leaves the device, no server round-trip.

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

## Install

**Zero-friction — prebuilt binaries, no Rust toolchain.** Installs both `mnema` (CLI) and
`mnema-mcp` (MCP server):

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/MerlijnW70/mnema/main/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/MerlijnW70/mnema/main/install.ps1 | iex
```

**From source, with a Rust toolchain** — builds and installs both binaries:

```bash
cargo install --git https://github.com/MerlijnW70/mnema mnema --features mcp
```

The store is created and encrypted on first use. You don't need to manage a key: **omit
`MNEMA_KEY`** and mnema generates a strong random per-store key file (`<store>.key`). Prefer a
portable passphrase (shared store, CI, an env-only secret)? Set `MNEMA_KEY` to any string — or to a
strong random one with `mnema keygen`:

```bash
export MNEMA_KEY="$(mnema keygen)"
```

Then [point your MCP client](#use-it-as-an-mcp-server) at `mnema-mcp`.

## Quickstart (from a checkout)

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

By default recall rides the zero-dependency lexical embedder. For **semantic** recall — matching
on meaning, not just shared words — build with the opt-in `local-embed` feature, which loads
`all-MiniLM-L6-v2` in-process (pure Rust, via candle; the model is fetched once and cached):

```bash
cargo build --release --features mcp,local-embed --bin mnema-mcp
```

The embedder is fixed at build time and a store is embedder-specific (the vector widths differ),
so keep one store per binary — a semantic binary opens the semantic store it wrote, and likewise
for the lexical one.

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

Tools: **remember** (with an `open` / `redacted` / `private` tier and an `importance`),
**recall**, **recent** (newest-first context, no query needed), **reinforce** (strengthen a
memory the agent actually used), **prune** (forget faded memories so the store stays bounded),
**remember_fact**, **beliefs**, **forget**, **stats**. The server is thin glue over the library
— every guarantee above still holds, and the memory logic stays mutation-pinned.

## Correctness

Every invariant above is pinned by a **zero-dependency mutation-coverage gate**, whose rule is
that *a green build proves the changed logic is tested* — a mutation to any pinned branch must
turn a test red, or the build is not green. At last check: **192/192 viable mutants killed,
0 survivors, 100%**.

## Validated in a live agentic loop

Beyond its unit tests, `mnema` has been driven by a **real self-improving agent**: the loop
stored each iteration's verdict as a memory and recalled the relevant lessons before the next
proposal. In live runs the proposer **demonstrably used the recall** — it cited a recorded
rejection by its detail code and redesigned its change specifically so it would *not* repeat the
remembered failure. Contradiction-resolving writes kept the belief base current across iterations;
the egress filter and encrypted store held throughout. The memory layer works not just in a test
harness, but as the actual memory of an agent doing real work.

## License

Dual-licensed under **MIT OR Apache-2.0**.
