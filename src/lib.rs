#![forbid(unsafe_code)]

//! `mnema` — the local-first LLM memory layer (`docs/proposals/mnema-memory-layer.md`).
//!
//! This is the **umbrella** crate. Its zero-dependency heart lives in the separate [`mnema_core`]
//! crate — the egress-filter privacy invariant (ADR-0021), the memory model, and the pure-logic
//! stores (`semantic`, `working`, `vector`, `embed`, `retrieval`). That crate is a strictly
//! zero-dependency, `unsafe`-free leaf, certified as such by `internal-tool gate` on every run; keeping it a
//! separate crate is what makes "the core takes no dependencies" a *mechanically enforced* fact rather
//! than a hope (ADR-0020). Everything that legitimately needs a dependency — encryption at rest, a real
//! embedder, JSON for the MCP server — is opt-in and lives here, one layer up.
//!
//! The core is re-exported flat, so `mnema::Memory`, `mnema::retrieval::…`, `mnema::vector::Embedder`
//! and friends resolve exactly as before the split.

pub use mnema_core::*;

/// Encrypted episodic store (slice 2). Behind the `secure` feature (ADR-0020) so the core stays
/// zero-dependency; only the full build compiles it.
#[cfg(feature = "secure")]
pub mod store;

/// Per-store key resolution ($MNEMA_KEY or a sidecar `<store>.key`) shared by the CLI and MCP
/// server. I/O glue below the behavioral waterline, behind `secure` (needs the encrypted store).
#[cfg(feature = "secure")]
pub mod keyfile;

/// The `Mnema` facade (Phase 2c): one `remember`/`recall`/`forget` API over the whole stack. Behind
/// the `secure` feature — it builds on the encrypted episodic store.
#[cfg(feature = "secure")]
pub mod facade;

/// A real in-process semantic `Embedder` (all-MiniLM-L6-v2 via candle), behind the opt-in
/// `local-embed` feature. Model glue below the behavioral waterline — its pooling math is in
/// [`mnema_core::embed`] and is probed; the transformer itself is not.
#[cfg(feature = "local-embed")]
pub mod model_embed;

/// A semantic `Embedder` backed by a local HTTP embeddings endpoint (Ollama / llama.cpp), behind the
/// opt-in `http-embed` feature — meaning-based recall with no bundled model. Umbrella-only I/O glue,
/// below the behavioral waterline.
#[cfg(feature = "http-embed")]
pub mod http_embed;
