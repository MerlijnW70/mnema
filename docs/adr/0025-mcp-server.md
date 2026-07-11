# ADR-0025 — The `mnema-mcp` server and its `serde_json` dependency

- **Status:** Accepted
- **Amends:** [ADR-0020](0020-mnema-dependency-budget.md) (dependency budget)

## Context

The single highest-leverage adoption surface for a local-first agent memory is a **Model
Context Protocol server**: any MCP client (Claude Code, Cursor, Claude Desktop) can then give
an agent private, local, encrypted memory with one line of config. MCP speaks **JSON-RPC 2.0**,
so the server must parse arbitrary JSON requests and serialise JSON responses (including nested
tool-call arguments and tool schemas).

Per ADR-0020, adding a crate is an amendment that must (1) enumerate it, (2) justify why a
hand-roll won't do, (3) confirm it is safe Rust, and (4) keep the change out of the probed
substrate.

## Decision

Add an optional **`mcp`** feature — `mcp = ["secure", "dep:serde_json"]` — and a `mnema-mcp`
binary (`src/bin/mnema-mcp.rs`) built only under it.

- **The crate:** `serde_json`. It is **already in the dependency tree** (enabled by
  `local-embed` and as a dev-dependency for the LoCoMo bench), pure safe Rust, and the de-facto
  standard for JSON in Rust. No *new* crate enters the tree.
- **Why not hand-roll:** a correct, robust JSON-RPC codec (arbitrary nested arguments in, tool
  schemas + results out) is a few hundred lines of fiddly glue with no correctness upside; the
  memory logic — where correctness matters — is elsewhere and already pinned.
- **Scope:** `serde_json` is used **only in the below-waterline binary**, never in the probed
  library (`embed`, `facade`, `lib`, `retrieval`, `semantic`, `store`, `vector`, `working`).
  The **zero-dependency core and the `secure` budget are unchanged**; a default `cargo build`
  pulls in nothing new.

## Consequences

- The MCP glue stays **thin**: it speaks JSON-RPC and persists the store, but every real
  decision (the egress wall, contradiction resolution, packing) lives in the mutation-pinned
  facade. No memory logic lives in the binary.
- `recall` runs against `Destination::Remote`, so the egress wall is **on by default** — a
  `Private` memory can never leave through the tool. (Verified end-to-end.)
- Persistence: the store path and passphrase come from `MNEMA_PATH` / `MNEMA_KEY`; the store is
  re-sealed after every mutating call. The format-version byte and embedder-width header
  (Phase 0) make that on-disk format forward-safe.
- Future MCP work (resources, prompts, streaming) stays within this feature and this
  binary — it never relaxes the core budget.
