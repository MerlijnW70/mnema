# mnema — architecture & design

The [README](README.md) is the fast path. This is the *why it works* — the guarantees, how they're
enforced, and how they're proven. If you're evaluating mnema for something that matters, start here.

## The Rust library

The CLI and MCP server are thin glue over one library type, `Mnema`:

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
| **Episodic** | [`store`](src/store.rs) | timestamped events; encrypted at rest (Argon2id + XChaCha20-Poly1305), tamper-evident, hard-deletable |
| **Semantic** | [`semantic`](src/semantic.rs) | facts with **contradiction resolution** — a newer belief supersedes, never accumulates |
| **Working** | [`working`](src/working.rs) | ephemeral scratchpad — TTL horizon + capacity cap; not persisted |
| **Procedural** | *(modeled as semantic facts)* | learned preferences, e.g. "answer in metric" |

## The guarantees

Each is a hard invariant, pinned by the mutation gate (below):

| Invariant | Where enforced |
|---|---|
| A `Private` memory's content never reaches a `Remote` bundle | `lib` — one `pack_bundle` egress choke point |
| A resolved contradiction never leaves two *live* beliefs | `semantic` |
| Encryption round-trips; a wrong key or tampered byte never yields plaintext | `store` |
| A forgotten memory is gone from state and every re-sealed blob; its id is never reused | `store` |
| At full probe, the approximate index equals the exact oracle | `vector` (`IvfIndex`) |
| A crash mid-write, or a wrong key, never destroys the store | CLI/MCP — atomic write + refuse-on-unopenable |

## The egress wall (privacy)

Every memory carries a tier: `Open`, `Redacted`, or `Private`. Every path that could send memory to
a model — query recall, recency recall, belief lookup — funnels through a **single** function,
`pack_bundle`, which applies the tier against the destination:

- `Open` → content is sent.
- `Redacted` → only a sanitized surface is sent (never the full content).
- `Private` bound for `Remote` → dropped, unconditionally.

Because there is exactly one choke point, "a private memory cannot leak to the cloud" is a property
of one auditable function, not a convention scattered across the codebase. Stored text is always
delivered to the model as **data, never as an instruction** — injection-resistant by construction.

## Encrypted at rest

`seal`/`open` derive a key from the passphrase with **Argon2id** (memory-hard) and encrypt the whole
store — episodic log, beliefs, clock — with **XChaCha20-Poly1305** (AEAD). The on-disk blob is
`version || salt || nonce || AEAD(plaintext)`; a wrong key or a single flipped byte fails the
authentication tag and yields an error, never plaintext. Writes are **atomic** (temp file + rename),
so a crash or full disk leaves the previous store intact. A wrong key on an existing store makes the
server **refuse to start** rather than overwrite it. Concurrent writers are serialized by an advisory
lock. Deletes are hard: `forget` removes matching events from state and every future re-sealed blob,
and their ids are never reissued.

Not claimed: unbreakable, or safe against a compromised OS / root / a malicious local model.
Local-first raises the bar; it doesn't make the machine a vault. `#![forbid(unsafe_code)]` crate-wide.

## Retrieval

Hybrid recall fuses three retrievers with **reciprocal-rank fusion**:

- **dense** — cosine over embeddings (exact `VectorIndex`, or approximate `IvfIndex`);
- **recency** — newest first;
- **keyword** — BM25 lexical overlap.

A **forgetting curve** then weights each hit by `importance × 0.5^(age / half_life)`, near-duplicates
are suppressed, and the result is packed through the egress filter under a character budget. Recall is
**exact** by default (the correctness oracle); an approximate `IvfIndex` trades recall for speed and is
opt-in (`build_ann` + `recall_fast`) — at full probe it returns exactly what the exact index does.

### Embedders

By default recall uses a zero-dependency lexical embedder (`HashEmbedder`) — no model, no download.
Build with the `local-embed` feature for real **semantic** recall (`all-MiniLM-L6-v2` via candle, pure
Rust; the model is fetched and cached on first use). The embedder is fixed per store — its vector width
is recorded, and opening a store with a mismatched embedder is refused.

## How the guarantees are proven

Every invariant is pinned by a **zero-dependency behavioral mutation-coverage gate**. Its rule: *a
green build proves the changed logic is tested* — mutate any pinned branch and a test must turn red,
or the build isn't green. Literals and exact boundaries (which the operator set can't reach) are
pinned separately by hand-computed value tests. The gate is complemented by a fuzz test over the store
parser (adversarial bytes never panic) and a multi-agent adversarial audit of the cross-surface
invariants.

## Runs in a browser

The zero-dependency core compiles to **WebAssembly** (`wasm32-unknown-unknown`), so the memory layer
can run entirely in a browser tab — private memory with no server round-trip.

## Validated in a live agentic loop

Beyond unit tests, mnema has been driven by a real self-improving agent: the loop stored each
iteration's verdict as a memory and recalled the relevant lessons before the next proposal. In live
runs the proposer demonstrably used the recall — it cited a recorded rejection by its detail code and
redesigned its change specifically so it would *not* repeat the remembered failure. The
contradiction-resolving writes, the egress filter, and the encrypted store all held throughout.

## Feature flags

| Feature | Enables |
|---|---|
| *(none)* | the zero-dependency core: retrieval, semantic, working, vector, egress |
| `secure` | encryption at rest (Argon2id + XChaCha20-Poly1305), the `Mnema` facade, the `mnema` CLI |
| `mcp` | the `mnema-mcp` server (implies `secure`) |
| `local-embed` | the bundled semantic embedder (all-MiniLM-L6-v2 via candle) |

## License

Dual-licensed under **MIT OR Apache-2.0**.
