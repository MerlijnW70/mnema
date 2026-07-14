# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0: minor = may break,
patch = additive or fixes).

## [0.1.4] - 2026-07-15

### Added
- **`forget_fact` MCP tool** — hard-delete beliefs about a subject (or one attribute), the belief
  equivalent of `forget`. Lets an agent correct or remove a wrong/stale belief over MCP.
- **`mnema-server --local` (or `$MNEMA_LOCAL`)** — switches recall/recent/beliefs to the local
  egress destination so an **on-device** model can read `private` memories. It is a deployment
  choice fixed at startup, never a per-call argument, so a caller can't open the egress wall itself.
  The default stays remote (private memories withheld).
- **CLI parity** — `recent`, `beliefs`, `reinforce`, `forget`, and `forget-fact` commands, over the
  same facade the MCP server uses. (Previously there was no way to delete a memory from the CLI.)

## [0.1.3] - 2026-07-14

### Fixed
- Pointing `--path` (or `$MNEMA_PATH`) at a not-yet-created directory no longer fails with a
  cryptic `cannot open lock file … path not found`. The CLI and server now create the store's
  parent directory (`create_dir_all`) at the lock gate, with a clear error if that fails.

## [0.1.2] - 2026-07-14

### Added
- **`mnema-core`** — the pure, **strictly zero-dependency** heart of the layer (the egress
  filter, the memory model, and the retrieval/semantic/vector/working stores) is now a separate
  crate. It compiles to `wasm32-unknown-unknown` and is certified a dependency-free leaf on every
  build. The umbrella `mnema` crate re-exports it, so the public API is unchanged; the
  dependency-carrying features (`secure`, `local-embed`, `mcp`) live one layer up.

### Fixed
- **Keyfile durability & atomicity** — the sidecar `<store>.key` is now written atomically,
  durably, and owner-only (temp file + `O_EXCL` + mode `0600` + fsync + rename + parent-directory
  fsync), and write/permission failures are surfaced instead of silently ignored. Previously the
  key was briefly world-readable and never fsynced, so a `rekey` interrupted by power loss could
  leave the re-sealed store unrecoverable.
- **Non-finite `importance` on the load path** — a `NaN`/`±inf` importance decoded from disk (or
  set via the low-level API) is now neutralized to `1.0` at the point of use, so it can no longer
  scramble the recall ordering or make a memory un-prunable. The earlier guard only covered the
  write path.
- **Durability of every persisted write** — `write_atomic` now fsyncs the parent directory after
  the rename, so an acknowledged write survives power loss (not just a clean process exit).
- **Egress before dedup** — near-duplicate suppression now runs *after* dropping egress-denied
  memories, so a `Private` memory can no longer shadow an emittable `Open` near-duplicate bound for
  a remote model.
- **`FactId` uniqueness** — belief ids are assigned as `max + 1` rather than `len()`, so they stay
  unique after a `forget`.

### Changed
- Derived keys and passphrases are wiped from memory on drop (`zeroize`; umbrella-only, under the
  `secure` feature).
- `salt_of` validates the format-version byte before the caller runs the memory-hard KDF.
- `StoreError` is now `#[non_exhaustive]` (the on-disk format will grow failure modes).
- Removed the `Default` derive on `VectorIndex` / `IvfIndex` (a default, `dims == 0` index was an
  unusable trap; construct with `::new` against a real width).

### Tests
- Added coverage for previously-untested glue: the MCP server's fail-closed tier parsing and the
  keyfile generate / round-trip / owner-only / refuse-existing-store / malformed paths.

## [0.1.1] - 2026-07-12

### Changed
- **Breaking:** renamed the server binary `mnema-mcp` → **`mnema-server`** and added a `--path`
  flag (falls back to `$MNEMA_PATH`, then `./mnema.store`) to choose the store location.
- Rewrote the README for zero-friction onboarding; moved the deep dive to `ARCHITECTURE.md`.
- Sharper crates.io description and keywords.

## [0.1.0] - 2026-07-12

Initial release: a local, encrypted memory layer for AI agents.

### Added
- **Privacy egress wall** — every memory carries an `Open` / `Redacted` / `Private` tier; a
  `Private` memory's content never enters a bundle bound for a remote model (ADR-0021).
- **Encryption at rest** — Argon2id + XChaCha20-Poly1305 seal/open of the whole store (`secure`).
- **Contradiction-resolving beliefs** — asserting a fact resolves against the live belief
  (reinforce / supersede / keep-as-history); at most one live belief per key.
- **Hybrid retrieval** — reciprocal-rank fusion of dense (vector), recency, and lexical (BM25)
  retrievers, a forgetting curve (`importance × decay`), near-duplicate suppression, and pruning
  of faded memories to keep a long-lived store bounded.
- **Optional semantic recall** — an in-process `all-MiniLM-L6-v2` embedder via candle
  (`local-embed`); a zero-dependency lexical embedder is the default.
- **MCP server** — `remember` / `recall` / `recent` / `remember_fact` / `beliefs` / `stats` /
  `forget` over an encrypted local store, for any MCP client.
- **Robust persistence** — atomic writes, an inter-process store lock, resumable `rekey`, a
  refuse-to-overwrite guard on an unopenable store, and a panic-free (fuzzed) decode path.
- **Zero-friction install** — prebuilt binaries via `curl | sh` / PowerShell, plus `mnema keygen`
  for a strong `$MNEMA_KEY` passphrase.

[0.1.4]: https://github.com/MerlijnW70/mnema/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/MerlijnW70/mnema/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MerlijnW70/mnema/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MerlijnW70/mnema/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MerlijnW70/mnema/releases/tag/v0.1.0
