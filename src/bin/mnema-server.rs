//! `mnema-server` — a Model Context Protocol server exposing mnema's memory as tools over stdio.
//!
//! Any MCP client (Claude Code, Cursor, Claude Desktop) can give an agent **private, local,
//! encrypted** memory by pointing it at this binary:
//!
//! ```jsonc
//! // e.g. in an MCP client config
//! { "command": "mnema-server", "args": ["--path", "~/mnema.store"] }
//! ```
//!
//! The store path is `--path`, else `$MNEMA_PATH`, else `./mnema.store`. Set `$MNEMA_KEY` to a
//! passphrase, or omit it for an auto-generated per-store key file.
//!
//! This is **below-waterline glue**: it speaks line-delimited JSON-RPC and persists the store,
//! but every real decision — the egress wall, contradiction resolution, packing — lives in the
//! mutation-pinned [`mnema`] facade. By default `recall` runs against `Destination::Remote`, so a
//! `Private` memory can never leave through this tool. Launching with `--local` (or `$MNEMA_LOCAL`)
//! switches recall to `Destination::Local` so an **on-device** model can read Private memories —
//! a deployment choice set at startup, never a per-call one, so a caller can't open the wall itself.
//!
//! ## Embedder
//!
//! By default recall rides the zero-dependency `HashEmbedder` (a lexical bag-of-tokens vector).
//! For meaning-based recall, build with one of: `--features mcp,http-embed` — embed via a local
//! HTTP endpoint (Ollama / llama.cpp; see `mnema::http_embed`), the light path; or
//! `--features mcp,local-embed` — an in-process `all-MiniLM-L6-v2` via candle, self-contained but a
//! heavier build. The choice is fixed at compile time, and a store is embedder-specific (the widths
//! differ), so a given binary always opens the store it wrote.

use std::io::{BufRead, Write};

#[cfg(not(any(feature = "local-embed", feature = "http-embed")))]
use mnema::embed::HashEmbedder;
use mnema::facade::Mnema;
use mnema::retrieval::RetrievalWeights;
use mnema::vector::Embedder;
use mnema::{BundleItem, Destination, EgressTier};
use serde_json::{Value, json};

/// The retriever-fusion weights recall uses. With a real semantic embedder (`local-embed` /
/// `http-embed`), tip toward the dense retriever so a meaning-match outvotes a mere keyword or
/// recency overlap; with the lexical default the dense signal is noise, so stay balanced.
#[cfg(any(feature = "local-embed", feature = "http-embed"))]
fn recall_weights() -> RetrievalWeights {
    RetrievalWeights::semantic()
}
#[cfg(not(any(feature = "local-embed", feature = "http-embed")))]
fn recall_weights() -> RetrievalWeights {
    RetrievalWeights::default()
}

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The server's launch configuration.
struct ServerConfig {
    /// The store path: `--path <file>` if given, else `$MNEMA_PATH`, else `mnema.store`.
    path: String,
    /// The egress destination applied to *every* recall for the whole session. `Remote` (the
    /// default) filters out Private memories — correct when this server feeds a cloud model.
    /// `--local` (or `$MNEMA_LOCAL`) sets `Local`, so recall may return Private memories — use it
    /// ONLY when the server feeds an on-device model. It is a **deployment** decision set at launch,
    /// never a per-call argument, so a caller/model can never flip the egress wall open itself.
    dest: Destination,
    /// `--migrate`: re-embed the existing store under this build's embedder and exit, instead of
    /// serving. Used to move a store to a different embedder (e.g. a lexical store to a semantic
    /// build) without losing data — otherwise the mismatched build refuses to open it.
    migrate: bool,
}

/// Parse `--path`, `--local`, `--migrate`, `--help`. An MCP client can pass
/// `args: ["--path", "…", "--local"]` or set `$MNEMA_PATH` / `$MNEMA_LOCAL` — either works.
fn resolve_config() -> ServerConfig {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path = None;
    let mut local = std::env::var("MNEMA_LOCAL")
        .map(|v| env_switch_on(&v))
        .unwrap_or(false);
    let mut migrate = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                i += 1;
                path = Some(args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("mnema-server: --path needs a file argument");
                    std::process::exit(2);
                }));
            }
            "--local" => local = true,
            "--migrate" => migrate = true,
            "-h" | "--help" => {
                println!(
                    "mnema-server — MCP memory server (stdio JSON-RPC)\n\n\
                     USAGE: mnema-server [--path <store>] [--local] [--migrate]\n\n\
                     Store path: --path, else $MNEMA_PATH, else ./mnema.store.\n\
                     --local (or $MNEMA_LOCAL=1): recall may return Private memories — set this ONLY\n\
                     when the server feeds an on-device model, never a cloud one.\n\
                     --migrate: re-embed the store under this build's embedder and exit (use once to\n\
                     move a store to a different embedder, e.g. lexical -> a semantic build).\n\
                     $MNEMA_KEY sets a passphrase; omit it for an auto-generated <store>.key."
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("mnema-server: unknown argument '{other}' (try --help)");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let path = path
        .or_else(|| std::env::var("MNEMA_PATH").ok())
        .unwrap_or_else(|| "mnema.store".to_string());
    let dest = if local {
        Destination::Local
    } else {
        Destination::Remote
    };
    ServerConfig {
        path,
        dest,
        migrate,
    }
}

fn main() {
    let cfg = resolve_config();
    // Hold an exclusive advisory lock on the store for the whole session. A resident server keeps
    // the store in RAM and re-seals it on every write, so a second concurrent writer's records
    // would be silently clobbered by the next re-seal. `_lock` lives to the end of main; the OS
    // releases it when the process exits (even on crash). Taken first, so a keyfile generated below
    // (for a fresh store) is written under the lock.
    let _lock = lock_store(&cfg.path);

    // Resolve the per-store key the same way the CLI does: $MNEMA_KEY, else a random sidecar
    // <store>.key (generated for a fresh store). This shares a store family with the CLI and never
    // seals under an empty passphrase — an unset key no longer means weak, silent encryption.
    let key = mnema::keyfile::resolve_key(std::path::Path::new(&cfg.path)).unwrap_or_else(|e| {
        eprintln!("mnema-server: {e}");
        std::process::exit(1);
    });

    // The embedder is chosen at compile time; `run` is generic over it. Precedence: an in-process
    // candle model (`local-embed`), else a local HTTP embeddings endpoint (`http-embed`), else the
    // zero-dependency lexical `HashEmbedder`.
    #[cfg(feature = "local-embed")]
    run(&cfg, &key, || {
        mnema::model_embed::MiniLmEmbedder::load().unwrap_or_else(|e| {
            eprintln!("mnema-server: could not load the semantic model ({e})");
            std::process::exit(1);
        })
    });
    #[cfg(all(feature = "http-embed", not(feature = "local-embed")))]
    run(&cfg, &key, || {
        mnema::http_embed::HttpEmbedder::from_env().unwrap_or_else(|e| {
            eprintln!(
                "mnema-server: could not reach the embeddings endpoint ({e}). Is your local \
                 embedding server running? Set $MNEMA_EMBED_URL / $MNEMA_EMBED_MODEL."
            );
            std::process::exit(1);
        })
    });
    #[cfg(not(any(feature = "local-embed", feature = "http-embed")))]
    run(&cfg, &key, || HashEmbedder::new(HashEmbedder::DEFAULT_DIMS));
}

/// With the chosen embedder: either migrate the store to it and exit (`--migrate`), or open the
/// store and serve. Generic over the embedder so all three feature builds share one path.
fn run<E: Embedder>(cfg: &ServerConfig, key: &[u8], make: impl Fn() -> E) {
    if cfg.migrate {
        migrate_and_exit(&cfg.path, key, make());
    }
    let store = open_store(&cfg.path, key, make);
    serve(store, &cfg.path, key, cfg.dest);
}

/// Re-embed the existing store under `embedder` (possibly a different width), re-seal, write, and
/// exit — the escape hatch when the store was written by a different embedder (e.g. a lexical store
/// opened by a semantic build). Requires an existing store; run once, then start the server normally.
fn migrate_and_exit<E: Embedder>(path: &str, key: &[u8], embedder: E) -> ! {
    let width = embedder.dims();
    let blob = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("mnema-server: nothing to migrate — cannot read {path} ({e})");
        std::process::exit(1);
    });
    let mut store = Mnema::migrate(&blob, key, embedder).unwrap_or_else(|e| {
        eprintln!("mnema-server: migrate failed ({e:?}) — wrong $MNEMA_KEY, or a corrupt store");
        std::process::exit(1);
    });
    let count = store.len();
    let out = store.seal(key).unwrap_or_else(|e| {
        eprintln!("mnema-server: migrate re-seal failed ({e:?})");
        std::process::exit(1);
    });
    write_atomic(path, &out).unwrap_or_else(|e| {
        eprintln!("mnema-server: migrate write failed: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "mnema-server: migrated {count} memories to a width-{width} embedder — {path} now opens \
         with this build."
    );
    std::process::exit(0);
}

/// The JSON-RPC read/dispatch/persist loop, generic over the embedder in play. `dest` is the
/// launch-time egress destination (see [`ServerConfig`]) applied to every recall.
fn serve<E: Embedder>(mut store: Mnema<E>, path: &str, key: &[u8], dest: Destination) {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    serve_io(&mut store, path, key, dest, stdin.lock(), &mut stdout);
}

/// One line's worth of stdin: a complete line, a line that blew the size cap, or end-of-input.
enum LineRead {
    Line(Vec<u8>),
    /// The line exceeded [`MAX_LINE_BYTES`] — its excess was drained so the stream stays framed.
    Oversized,
    Eof,
}

/// A single JSON-RPC message may not exceed this many bytes. Reading a line unbounded would let
/// one hostile/broken client message grow the server's memory without limit; past the cap the
/// rest of the line is drained (so framing survives) and the message is rejected as a parse error.
const MAX_LINE_BYTES: u64 = 16 * 1024 * 1024;

/// Read one `\n`-terminated line, never buffering more than [`MAX_LINE_BYTES`] of it.
fn read_line_capped<R: BufRead>(reader: &mut R) -> std::io::Result<LineRead> {
    let mut buf = Vec::new();
    // `Read::take` consumes its receiver, so hand it a reborrow (`&mut *reader`), not the
    // caller's reference itself.
    let n = std::io::Read::take(&mut *reader, MAX_LINE_BYTES).read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(LineRead::Eof);
    }
    if buf.last() != Some(&b'\n') && n as u64 == MAX_LINE_BYTES {
        // Cap hit mid-line: drain (and discard) the oversized line's remainder byte-by-byte up
        // to its newline, so the NEXT message starts cleanly framed. (A 16 MiB+ single message
        // is a protocol violation; it gets a parse error, the session survives.)
        let mut byte = [0u8; 1];
        loop {
            let m = reader.read(&mut byte)?;
            if m == 0 || byte[0] == b'\n' {
                break;
            }
        }
        return Ok(LineRead::Oversized);
    }
    Ok(LineRead::Line(buf))
}

/// The transport-agnostic serve loop: `reader` feeds line-delimited JSON-RPC, replies go to
/// `writer`. Factored off stdin/stdout so the full dispatch — framing, notification semantics,
/// error codes, the egress wall — is exercisable in tests.
fn serve_io<E: Embedder>(
    store: &mut Mnema<E>,
    path: &str,
    key: &[u8],
    dest: Destination,
    mut reader: impl BufRead,
    writer: &mut impl Write,
) {
    loop {
        let raw = match read_line_capped(&mut reader) {
            Ok(LineRead::Eof) => break,
            Ok(LineRead::Oversized) => {
                write_msg(
                    writer,
                    &error(Value::Null, -32700, "parse error: message too large"),
                );
                continue;
            }
            Ok(LineRead::Line(b)) => b,
            // A transport read error is end-of-input: nothing more can arrive on a broken pipe.
            Err(_) => break,
        };
        // Validate UTF-8 ourselves (read_until is byte-level): a non-UTF-8 line is reported and
        // skipped, so one bad byte can't silently kill the whole session.
        let Ok(line) = std::str::from_utf8(&raw) else {
            write_msg(
                writer,
                &error(Value::Null, -32700, "parse error: non-UTF-8 input"),
            );
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(line) else {
            write_msg(writer, &error(Value::Null, -32700, "parse error"));
            continue;
        };
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str);
        // Dispatch to a JSON-RPC outcome. `Ok` is a result payload; `Err` is (code, message).
        let outcome: Result<Value, (i64, String)> = match method {
            // No/invalid "method" member is an invalid Request (-32600), not "method not found".
            None => Err((-32600, "invalid request: no method".to_string())),
            Some("initialize") => Ok(initialize_result()),
            Some("tools/list") => Ok(tools_list()),
            Some("tools/call") => Ok(handle_tool_call(store, path, key, dest, req.get("params"))),
            Some("resources/list") => Ok(resources_list()),
            Some("resources/read") => {
                handle_resources_read(store, dest, req.get("params")).map_err(|m| (-32602, m))
            }
            Some("prompts/list") => Ok(prompts_list()),
            Some("prompts/get") => {
                handle_prompts_get(store, dest, req.get("params")).map_err(|m| (-32602, m))
            }
            Some("ping") => Ok(json!({})),
            Some(m) if m.starts_with("notifications/") => continue,
            Some(_) => Err((-32601, "method not found".to_string())),
        };
        // A message without an id is a JSON-RPC *notification*: its side effects (a tools/call
        // above still ran) happen, but the server MUST NOT reply — not even with id:null.
        let Some(id) = id else { continue };
        match outcome {
            Ok(result) => write_msg(writer, &reply(id, result)),
            Err((code, msg)) => write_msg(writer, &error(id, code, &msg)),
        }
    }
}

fn write_msg(out: &mut impl Write, msg: &Value) {
    let _ = writeln!(out, "{msg}");
    let _ = out.flush();
}

fn reply(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        // Tools (remember/recall/…), plus a `recent` **resource** a client can auto-load as
        // session-start context, and a `recall` **prompt** for on-demand memory injection.
        "capabilities": { "tools": {}, "resources": {}, "prompts": {} },
        "serverInfo": { "name": "mnema", "version": SERVER_VERSION }
    })
}

/// The one resource: the recent memories a client can load as ambient context at session start,
/// without the agent having to decide to call `recall`. Egress-filtered like every read.
const RECENT_URI: &str = "mnema://recent";

fn resources_list() -> Value {
    json!({ "resources": [ {
        "uri": RECENT_URI,
        "name": "Recent memories",
        "description": "The most recent stored memories — load at session start so the agent knows what it already learned. Private memories are filtered out.",
        "mimeType": "text/plain"
    } ] })
}

/// Handle `resources/read`: only [`RECENT_URI`] is served, rendered as newest-first text.
/// An unknown uri is `Err` — a JSON-RPC error at the transport, not a tools/call-shaped
/// `isError` result, which a resources client has no reason to understand.
fn handle_resources_read<E: Embedder>(
    store: &Mnema<E>,
    dest: Destination,
    params: Option<&Value>,
) -> Result<Value, String> {
    let uri = params
        .and_then(|p| p.get("uri"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if uri != RECENT_URI {
        return Err(format!("unknown resource: {uri:?}"));
    }
    let items = store.recall_recent(dest, 4000);
    let text = if items.is_empty() {
        "(no memories yet)".to_string()
    } else {
        render_items(&items)
    };
    Ok(json!({ "contents": [ { "uri": RECENT_URI, "mimeType": "text/plain", "text": text } ] }))
}

fn prompts_list() -> Value {
    json!({ "prompts": [ {
        "name": "recall",
        "description": "Pull the memories relevant to a query into the conversation, so past context informs this turn.",
        "arguments": [ { "name": "query", "description": "what to recall about", "required": true } ]
    } ] })
}

/// Handle `prompts/get`: the `recall` prompt returns the recalled memories as a user message.
/// An unknown prompt name is `Err` — a JSON-RPC error, mirroring [`handle_resources_read`].
fn handle_prompts_get<E: Embedder>(
    store: &Mnema<E>,
    dest: Destination,
    params: Option<&Value>,
) -> Result<Value, String> {
    let name = params
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if name != "recall" {
        return Err(format!("unknown prompt: {name:?}"));
    }
    let query = params
        .and_then(|p| p.get("arguments"))
        .and_then(|a| a.get("query"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut hits = store.recall_decayed_weighted(query, dest, 5, 2000, 0, recall_weights());
    // The facade's k caps each retriever, not their fused union — cap the returned list itself.
    hits.truncate(5);
    let body = if hits.is_empty() {
        format!("(no memories relevant to {query:?})")
    } else {
        format!("Relevant memories for {query:?}:\n{}", render_items(&hits))
    };
    Ok(json!({
        "description": "Recalled memories",
        "messages": [ { "role": "user", "content": { "type": "text", "text": body } } ]
    }))
}

fn tools_list() -> Value {
    json!({ "tools": [
        {
            "name": "remember",
            "description": "Store a memory. 'tier' controls egress: a 'private' memory is never returned to a remote/cloud model. 'importance' makes a memory rank higher in recall and resist forgetting. For a 'redacted' memory, 'redacted' is the sanitized surface a remote model sees in place of the full content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "the memory text to store" },
                    "tier": { "type": "string", "enum": ["open", "redacted", "private"], "description": "egress tier (default: open)" },
                    "importance": { "type": "number", "description": "how salient this memory is (default 1.0); higher ranks higher in recall" },
                    "redacted": { "type": "string", "description": "for a 'redacted' memory, the sanitized text a remote model sees instead of the content (a redacted memory with no surface reveals nothing remotely)" }
                },
                "required": ["content"]
            }
        },
        {
            "name": "recall",
            "description": "Retrieve the most relevant stored memories for a query. Private memories are filtered out — they never reach a remote model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "k": { "type": "integer", "description": "max memories to return (default 5)" },
                    "budget": { "type": "integer", "description": "max total characters (default 2000)" },
                    "half_life": { "type": "integer", "description": "bias toward recent memories: salience halves every this-many stored memories of age (default 0 = importance only, no recency decay)" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "recent",
            "description": "List the most recently stored memories, newest first — the context an agent wants at the start of a session, no query needed. Private memories are filtered out.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k": { "type": "integer", "description": "max memories to return (default 5)" },
                    "budget": { "type": "integer", "description": "max total characters (default 2000)" }
                }
            }
        },
        {
            "name": "remember_fact",
            "description": "Store a belief as (subject, attribute, value), optionally at a tier — a 'private' belief is never returned to a remote/cloud model. A newer value supersedes an older contradicting one.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string" },
                    "attribute": { "type": "string" },
                    "value": { "type": "string" },
                    "tier": { "type": "string", "enum": ["open", "redacted", "private"], "description": "egress tier (default: open)" }
                },
                "required": ["subject", "attribute", "value"]
            }
        },
        {
            "name": "beliefs",
            "description": "List everything known about a subject — its live beliefs as 'subject.attribute = value'. Private beliefs are filtered out.",
            "inputSchema": {
                "type": "object",
                "properties": { "subject": { "type": "string" } },
                "required": ["subject"]
            }
        },
        {
            "name": "reinforce",
            "description": "Strengthen a memory you found useful (by the id shown in a recall result) so it ranks higher next time and resists forgetting.",
            "inputSchema": {
                "type": "object",
                "properties": { "id": { "type": "integer", "description": "the memory id from a recall result" } },
                "required": ["id"]
            }
        },
        {
            "name": "prune",
            "description": "Forget memories that have faded: those whose importance, decayed over 'half_life' ticks of age, has dropped below 'threshold'. Keeps a long-lived store bounded — reinforcement keeps what's used, this sheds what isn't. Destructive.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "half_life": { "type": "integer", "description": "ticks of age over which a memory's salience halves; 0 disables decay and prunes purely by importance" },
                    "threshold": { "type": "number", "description": "salience cutoff; a memory below this is forgotten (a memory at or above it is kept)" }
                },
                "required": ["half_life", "threshold"]
            }
        },
        {
            "name": "forget",
            "description": "Hard-delete every memory whose content contains the given substring.",
            "inputSchema": {
                "type": "object",
                "properties": { "contains": { "type": "string" } },
                "required": ["contains"]
            }
        },
        {
            "name": "forget_fact",
            "description": "Hard-delete beliefs about a subject — all of them, or only one attribute if 'attribute' is given. Use this to correct or remove a wrong or stale belief (the belief equivalent of 'forget').",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string" },
                    "attribute": { "type": "string", "description": "optional; if omitted, every belief about the subject is removed" }
                },
                "required": ["subject"]
            }
        },
        {
            "name": "stats",
            "description": "A privacy census of the store: memory and belief counts broken down by egress tier (how much is open vs. private) — counts only, no content.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ]})
}

fn tool_text(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

/// A tool result flagged as an error, so the agent sees the call failed rather than reading a
/// failure string as a normal answer.
fn tool_error(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": true })
}

/// Render a bundle as one `[id] text` line per item — the shared shape for `recall` and `recent`.
fn render_items(items: &[BundleItem]) -> String {
    items
        .iter()
        .map(|b| format!("[{}] {}", b.id, b.text))
        .collect::<Vec<_>>()
        .join("\n")
}

fn handle_tool_call<E: Embedder>(
    store: &mut Mnema<E>,
    path: &str,
    key: &[u8],
    dest: Destination,
    params: Option<&Value>,
) -> Value {
    let params = params.cloned().unwrap_or(Value::Null);
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = params.get("arguments").cloned().unwrap_or(Value::Null);
    let arg_str = |k: &str| {
        args.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    // A schema-required string argument: absent or non-string is an error the agent must SEE —
    // coercing to "" and reporting success would store an empty memory/belief and teach the
    // agent its (malformed) call worked.
    let required_str = |k: &str| -> Result<String, Value> {
        match args.get(k).and_then(Value::as_str) {
            Some(s) => Ok(s.to_string()),
            None => Err(tool_error(format!(
                "missing required string argument '{k}'"
            ))),
        }
    };

    let text = match name {
        "remember" => {
            let content = match required_str("content") {
                Ok(c) => c,
                Err(e) => return e,
            };
            let tier = match parse_tier(args.get("tier").and_then(Value::as_str)) {
                Ok(t) => t,
                Err(e) => return tool_error(e),
            };
            let importance = args
                .get("importance")
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32;
            // The redacted surface only bites on the redacted tier; empty for open/private.
            let redacted = arg_str("redacted");
            let id = store.remember_with(tier, importance, &content, &redacted);
            if let Err(e) = persist(store, path, key) {
                return tool_error(format!(
                    "memory {id} is in RAM but was NOT saved to disk: {e}"
                ));
            }
            format!("remembered as memory {id}")
        }
        "recall" => {
            let query = match required_str("query") {
                Ok(q) => q,
                Err(e) => return e,
            };
            let k = args.get("k").and_then(Value::as_u64).unwrap_or(5) as usize;
            let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2000) as usize;
            // half_life 0 keeps importance weighting (a memory marked important ranks higher)
            // without time decay; a positive value also biases toward recent memories.
            // Destination::Remote drops Private memories at the egress wall either way.
            let half_life = args.get("half_life").and_then(Value::as_u64).unwrap_or(0);
            let mut hits =
                store.recall_decayed_weighted(&query, dest, k, budget, half_life, recall_weights());
            // The tool contract is "k: max memories to RETURN". The facade's k caps each of its
            // fused retrievers (lexical/dense/recency), so their union can reach 3k — cap the
            // returned list itself.
            hits.truncate(k);
            if hits.is_empty() {
                "(no relevant memories)".to_string()
            } else {
                render_items(&hits)
            }
        }
        "recent" => {
            let k = args.get("k").and_then(Value::as_u64).unwrap_or(5) as usize;
            let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2000) as usize;
            // recall_recent is newest-first and egress-filtered (Private never leaves), so
            // taking the first k yields the k most recent shareable memories within budget.
            let mut items = store.recall_recent(dest, budget);
            items.truncate(k);
            if items.is_empty() {
                "(no memories yet)".to_string()
            } else {
                render_items(&items)
            }
        }
        "remember_fact" => {
            let (s, a, v) = match (
                required_str("subject"),
                required_str("attribute"),
                required_str("value"),
            ) {
                (Ok(s), Ok(a), Ok(v)) => (s, a, v),
                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return e,
            };
            let tier = match parse_tier(args.get("tier").and_then(Value::as_str)) {
                Ok(t) => t,
                Err(e) => return tool_error(e),
            };
            let res = store.remember_fact_tiered(&s, &a, &v, tier);
            if let Err(e) = persist(store, path, key) {
                return tool_error(format!("belief {s}.{a} is in RAM but was NOT saved: {e}"));
            }
            // Don't echo a private value back, even in the storage confirmation.
            let shown = if tier == EgressTier::Private {
                "<private>".to_string()
            } else {
                format!("{v:?}")
            };
            format!("belief {s}.{a} = {shown} ({res:?})")
        }
        "beliefs" => {
            let subject = match required_str("subject") {
                Ok(s) => s,
                Err(e) => return e,
            };
            // The launch-time `dest` applies the egress wall: with the default Remote, Private
            // beliefs are withheld; a `--local` server may surface them.
            let facts = store.beliefs(&subject, dest);
            if facts.is_empty() {
                format!("(nothing known about {subject})")
            } else {
                facts
                    .iter()
                    .map(|f| format!("{}.{} = {}", f.subject, f.attribute, f.value))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        "forget_fact" => {
            let subject = arg_str("subject");
            let attribute = arg_str("attribute");
            if subject.is_empty() {
                return tool_error("forget_fact needs a 'subject'".to_string());
            }
            // Delete every belief for the subject, or only the given attribute if one is supplied.
            // Hard delete (live + superseded), mirroring episodic `forget`.
            let removed = store.forget_facts(|f| {
                f.subject == subject && (attribute.is_empty() || f.attribute == attribute)
            });
            if let Err(e) = persist(store, path, key) {
                return tool_error(format!("forgot beliefs in RAM but did NOT save: {e}"));
            }
            let what = if attribute.is_empty() {
                subject.clone()
            } else {
                format!("{subject}.{attribute}")
            };
            format!("forgot {removed} belief record(s) for {what}")
        }
        "reinforce" => {
            let id = args.get("id").and_then(Value::as_u64).unwrap_or(u64::MAX);
            if store.reinforce(id) {
                if let Err(e) = persist(store, path, key) {
                    return tool_error(format!("reinforced memory {id} but did NOT save: {e}"));
                }
                format!("reinforced memory {id}")
            } else {
                format!("no memory with id {id}")
            }
        }
        "prune" => {
            let half_life = args.get("half_life").and_then(Value::as_u64).unwrap_or(0);
            let threshold = args.get("threshold").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let receipt = store.prune_faded(half_life, threshold);
            if let Err(e) = persist(store, path, key) {
                return tool_error(format!("pruned in RAM but did NOT save: {e}"));
            }
            format!(
                "pruned {} faded memories; {} remain",
                receipt.purged.len(),
                receipt.remaining
            )
        }
        "forget" => {
            let needle = match required_str("contains") {
                Ok(n) => n,
                Err(e) => return e,
            };
            let receipt = store.forget(|m| !needle.is_empty() && m.content.contains(&needle));
            if let Err(e) = persist(store, path, key) {
                return tool_error(format!("forgot in RAM but did NOT save: {e}"));
            }
            format!(
                "forgot {} memories; {} remain",
                receipt.purged.len(),
                receipt.remaining
            )
        }
        "stats" => {
            let s = store.stats();
            format!(
                "{} memories ({} open, {} redacted, {} private); {} live beliefs ({} private); {} indexed",
                s.total, s.open, s.redacted, s.private, s.beliefs, s.private_beliefs, s.indexed
            )
        }
        other => return tool_error(format!("unknown tool: {other}")),
    };
    tool_text(text)
}

/// Parse an environment switch **fail-closed**: only an explicit affirmative — `1`, `true`,
/// `yes`, `on` (case-insensitive, trimmed) — turns it on. This guards the egress wall itself:
/// `$MNEMA_LOCAL` decides whether Private memories may be recalled, so a falsy-but-non-empty
/// value (`MNEMA_LOCAL=0`, `=false`, `=off` — an operator being *explicit* that local mode is
/// OFF) must read as off, never silently open Private recall to a cloud model. The same
/// rejected-not-coerced discipline as [`parse_tier`].
fn env_switch_on(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Resolve the egress tier from a tool argument. Absent → `Open` (the documented default), but an
/// **unrecognised** tier string is rejected, not silently coerced: failing open to `Open` would let
/// a caller's typo (`"Private"`, `"priv"`) store an intended-private memory at the shareable tier,
/// which the next `Destination::Remote` recall would then leak. Fail closed, like the CLI's `tier`.
fn parse_tier(s: Option<&str>) -> Result<EgressTier, String> {
    match s {
        None | Some("open") => Ok(EgressTier::Open),
        Some("private") => Ok(EgressTier::Private),
        Some("redacted") => Ok(EgressTier::Redacted),
        Some(other) => Err(format!(
            "unknown tier {other:?} — use \"open\", \"redacted\", or \"private\""
        )),
    }
}

/// Seal and durably write the store. Returns an error message on failure so the caller can tell the
/// agent the write did NOT stick — reporting success on a failed persist would make an agent believe
/// a memory is saved when it is only in RAM and gone on restart.
fn persist<E: Embedder>(store: &mut Mnema<E>, path: &str, key: &[u8]) -> Result<(), String> {
    let blob = store
        .seal(key)
        .map_err(|e| format!("failed to seal store: {e:?}"))?;
    write_atomic(path, &blob).map_err(|e| format!("failed to write store to {path}: {e}"))
}

/// Write `bytes` to `path` durably: write a sibling `.tmp`, flush it to disk, then rename it
/// over `path`. The rename is atomic within the directory, so `path` is always either the whole
/// old store or the whole new one — a crash or full disk mid-write can never truncate it to a
/// torn blob that would then fail to open. On any failure the original `path` is left untouched.
fn write_atomic(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = format!("{path}.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    // Durability: a rename's directory entry is not on stable storage until the parent directory
    // is fsynced (POSIX). Best-effort — the rename is already atomic, so skipping this never
    // corrupts the store; it only weakens the "an acknowledged write survives power loss"
    // guarantee, so a dir-fsync failure doesn't fail the write.
    #[cfg(unix)]
    if let Some(dir) = std::path::Path::new(path)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        && let Ok(d) = std::fs::File::open(dir)
    {
        let _ = d.sync_all();
    }
    // Windows has no std way to open a directory handle (that needs FILE_FLAG_BACKUP_SEMANTICS),
    // so flush the renamed file itself: FlushFileBuffers on it forces its NTFS metadata — the
    // record the rename rewrote — to stable storage. Best-effort, like the unix dir-fsync.
    #[cfg(windows)]
    if let Ok(f) = std::fs::File::open(path) {
        let _ = f.sync_all();
    }
    Ok(())
}

/// Create the directory the store lives in, if it doesn't exist yet, so a `--path` pointing into a
/// not-yet-created folder just works instead of failing later with a cryptic "cannot open lock
/// file … path not found". Idempotent; a bare filename (no parent) is a no-op.
fn ensure_parent_dir(path: &str) {
    if let Some(dir) = std::path::Path::new(path)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        eprintln!(
            "mnema-server: cannot create store directory {} ({e})",
            dir.display()
        );
        std::process::exit(1);
    }
}

/// Take an exclusive advisory lock for the store via a sibling `<path>.lock`, returning the held
/// `File` (drop it to release; the OS also releases on process exit). Refuses to start if another
/// mnema process already holds it, so two writers can't clobber each other. The lock file is
/// separate from the store because the store is atomically *renamed* on every write, which would
/// drop a lock held on the store file itself.
fn lock_store(path: &str) -> std::fs::File {
    ensure_parent_dir(path);
    let lockpath = format!("{path}.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lockpath)
        .unwrap_or_else(|e| {
            eprintln!("mnema-server: cannot open lock file {lockpath} ({e})");
            std::process::exit(1);
        });
    if file.try_lock().is_err() {
        eprintln!(
            "mnema-server: {path} is already in use by another mnema process — refusing to start so a \
             concurrent writer's memories are not overwritten. Stop the other process and retry."
        );
        std::process::exit(1);
    }
    file
}

/// Open the store at `path`, or start a fresh one **only if there is no file yet**. `make`
/// builds the embedder; it is a factory (not a value) so the fresh-start branch can build one
/// after `open` has consumed the first — a model-backed embedder is not `Clone`.
///
/// Critically: if the file *exists* but will not open (wrong `MNEMA_KEY`, a corrupt blob, or a
/// newer on-disk format), we **refuse to start** rather than begin empty. Beginning empty would
/// let the next [`persist`] overwrite — and destroy — the real store; for a memory product, a
/// mistyped key must never mean silent data loss. The user fixes the key or moves the file.
fn open_store<E: Embedder>(path: &str, key: &[u8], make: impl Fn() -> E) -> Mnema<E> {
    match std::fs::read(path) {
        Ok(blob) => match Mnema::open(&blob, key, make()) {
            Ok(m) => m,
            // A width mismatch means the store was written by a *different embedder* (not a bad
            // key) — point the user at the migration escape hatch instead of "fix your key".
            Err(mnema::store::StoreError::EmbedderWidthMismatch { stored, embedder }) => {
                eprintln!(
                    "mnema-server: {path} was written by a different embedder (vector width {stored}, \
                     this build uses {embedder}). Re-embed it once with \
                     `mnema-server --migrate --path {path}`, then start normally."
                );
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!(
                    "mnema-server: {path} exists but could not be opened ({e:?}) — wrong MNEMA_KEY \
                     or a corrupt/newer store. Refusing to start so your memory is not \
                     overwritten. Fix the key, or move the file aside to begin fresh."
                );
                std::process::exit(1);
            }
        },
        // ONLY a genuinely absent file means "start fresh". Any other read error — a permission
        // or sharing violation (common on Windows: AV, backup, a second instance), a transient
        // I/O fault — must NOT start empty, or the next persist() would overwrite the real store
        // that we simply failed to read. Refuse, exactly like the unopenable case.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Mnema::new(make()),
        Err(e) => {
            eprintln!(
                "mnema-server: could not read {path} ({e}) — refusing to start so an existing store \
                 is not overwritten by an empty one. Resolve the I/O error (or move the file) and retry."
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tier_defaults_to_open_and_maps_known_tiers() {
        assert_eq!(parse_tier(None), Ok(EgressTier::Open));
        assert_eq!(parse_tier(Some("open")), Ok(EgressTier::Open));
        assert_eq!(parse_tier(Some("redacted")), Ok(EgressTier::Redacted));
        assert_eq!(parse_tier(Some("private")), Ok(EgressTier::Private));
    }

    #[test]
    fn parse_tier_fails_closed_on_an_unrecognized_tier() {
        // The security-load-bearing branch: a typo must be REJECTED, never silently coerced to
        // Open — coercing an intended-Private memory to Open would leak it on the next Remote
        // recall. Case matters too ("Private" is not "private").
        for bad in ["Private", "priv", "secret", "PRIVATE", "", "opened"] {
            assert!(
                parse_tier(Some(bad)).is_err(),
                "tier {bad:?} must be rejected, not coerced to a shareable tier"
            );
        }
    }

    #[test]
    fn recall_honors_the_launch_destination_for_private_memories() {
        use mnema::embed::HashEmbedder;
        use mnema::facade::Mnema;
        // The egress wall is set by the server's launch destination, never by the caller. A
        // default (Remote) server must withhold a Private memory; a --local server may surface it.
        let mut store = Mnema::new(HashEmbedder::new(HashEmbedder::DEFAULT_DIMS));
        store.remember(EgressTier::Open, "public trip to japan");
        store.remember(EgressTier::Private, "secret sk-live key");
        let call = json!({"name": "recall", "arguments": {"query": "japan key secret", "k": 10}});

        // recall is read-only (no persist), so path/key are unused here.
        let remote = handle_tool_call(
            &mut store,
            "unused",
            b"unused",
            Destination::Remote,
            Some(&call),
        );
        let rtext = remote["content"][0]["text"].as_str().unwrap();
        assert!(
            !rtext.contains("sk-live"),
            "Remote recall must never leak a Private memory: {rtext}"
        );
        assert!(rtext.contains("japan"));

        let local = handle_tool_call(
            &mut store,
            "unused",
            b"unused",
            Destination::Local,
            Some(&call),
        );
        let ltext = local["content"][0]["text"].as_str().unwrap();
        assert!(
            ltext.contains("sk-live"),
            "a --local server may surface a Private memory: {ltext}"
        );
    }

    #[test]
    fn resources_and_prompts_honor_the_egress_wall() {
        use mnema::embed::HashEmbedder;
        use mnema::facade::Mnema;
        let mut store = Mnema::new(HashEmbedder::new(HashEmbedder::DEFAULT_DIMS));
        store.remember(EgressTier::Open, "user ships rust");
        store.remember(EgressTier::Private, "secret sk-live key");

        // The `recent` resource is egress-filtered like recall: Remote withholds Private, Local may show it.
        let read = json!({ "uri": RECENT_URI });
        let remote = handle_resources_read(&store, Destination::Remote, Some(&read)).unwrap();
        let rt = remote["contents"][0]["text"].as_str().unwrap();
        assert!(
            !rt.contains("sk-live"),
            "resource must not leak a Private memory remotely: {rt}"
        );
        assert!(rt.contains("rust"));
        let local = handle_resources_read(&store, Destination::Local, Some(&read)).unwrap();
        assert!(
            local["contents"][0]["text"]
                .as_str()
                .unwrap()
                .contains("sk-live")
        );

        // The `recall` prompt is filtered too.
        let get = json!({ "name": "recall", "arguments": { "query": "secret key rust" } });
        let p = handle_prompts_get(&store, Destination::Remote, Some(&get)).unwrap();
        let pt = p["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(
            !pt.contains("sk-live"),
            "prompt must not leak a Private memory remotely: {pt}"
        );

        // Unknown uri / prompt name are transport-level errors (JSON-RPC error responses),
        // not tools/call-shaped `isError` results and not silent empties.
        assert!(
            handle_resources_read(
                &store,
                Destination::Remote,
                Some(&json!({"uri": "mnema://nope"}))
            )
            .is_err()
        );
        assert!(
            handle_prompts_get(&store, Destination::Remote, Some(&json!({"name": "nope"})))
                .is_err()
        );
    }

    /// Drive one line-delimited JSON-RPC session through `serve_io` in memory, returning the
    /// server's reply lines. The transport-level harness the dispatch fixes are pinned by.
    fn drive(input: &str) -> Vec<Value> {
        use mnema::embed::HashEmbedder;
        use mnema::facade::Mnema;
        let mut store = Mnema::new(HashEmbedder::new(HashEmbedder::DEFAULT_DIMS));
        let mut out: Vec<u8> = Vec::new();
        // path/key are only touched by persisting tools; the tests below use read-only methods
        // or accept the persist error text (the path is unwritable on purpose).
        serve_io(
            &mut store,
            "",
            b"k",
            Destination::Remote,
            input.as_bytes(),
            &mut out,
        );
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn env_switch_parses_fail_closed() {
        // The egress wall's own switch: ONLY an explicit affirmative may open Private recall.
        for on in ["1", "true", "TRUE", "yes", "on", " 1 ", "On"] {
            assert!(env_switch_on(on), "{on:?} must enable local mode");
        }
        // Falsy-but-non-empty values — an operator being explicit that local is OFF — and
        // anything unrecognized must stay OFF. `MNEMA_LOCAL=0` must never leak Private memories.
        for off in ["0", "false", "no", "off", "", " ", "2", "enable", "local"] {
            assert!(!env_switch_on(off), "{off:?} must NOT enable local mode");
        }
    }

    #[test]
    fn notifications_get_no_reply_and_requests_echo_their_id() {
        let replies = drive(concat!(
            r#"{"jsonrpc":"2.0","method":"ping"}"#,
            "\n", // notification: silence
            r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#,
            "\n", // request: one reply, id 7
            r#"{"jsonrpc":"2.0","id":"s","method":"tools/list"}"#,
            "\n", // string id echoes
        ));
        assert_eq!(
            replies.len(),
            2,
            "a notification (no id) must get NO reply — not even id:null: {replies:?}"
        );
        assert_eq!(replies[0]["id"], json!(7));
        assert_eq!(replies[1]["id"], json!("s"));
    }

    #[test]
    fn dispatch_distinguishes_invalid_request_from_method_not_found() {
        let replies = drive(concat!(
            r#"{"jsonrpc":"2.0","id":1}"#,
            "\n", // no method member
            r#"{"jsonrpc":"2.0","id":2,"method":"no/such"}"#,
            "\n", // unknown method
            r#"{"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"mnema://nope"}}"#,
            "\n",
            "not json\n",
        ));
        assert_eq!(replies[0]["error"]["code"], json!(-32600));
        assert_eq!(replies[1]["error"]["code"], json!(-32601));
        // An unknown resource is a JSON-RPC error (invalid params), not a result payload.
        assert_eq!(replies[2]["error"]["code"], json!(-32602));
        assert!(replies[2].get("result").is_none());
        assert_eq!(replies[3]["error"]["code"], json!(-32700));
    }

    #[test]
    fn an_oversized_line_is_rejected_and_the_session_survives() {
        // A line past MAX_LINE_BYTES is drained and rejected as a parse error; the next,
        // well-formed message on the same session still gets served.
        let mut input = vec![b'x'; (MAX_LINE_BYTES + 10) as usize];
        input.push(b'\n');
        input.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}\n");
        use mnema::embed::HashEmbedder;
        use mnema::facade::Mnema;
        let mut store = Mnema::new(HashEmbedder::new(HashEmbedder::DEFAULT_DIMS));
        let mut out: Vec<u8> = Vec::new();
        serve_io(
            &mut store,
            "",
            b"k",
            Destination::Remote,
            input.as_slice(),
            &mut out,
        );
        let replies: Vec<Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(replies.len(), 2, "oversized error + the ping reply");
        assert_eq!(replies[0]["error"]["code"], json!(-32700));
        assert_eq!(replies[1]["id"], json!(9));
        assert!(replies[1].get("result").is_some());
    }

    #[test]
    fn required_string_arguments_are_rejected_when_missing_not_coerced_to_empty() {
        use mnema::embed::HashEmbedder;
        use mnema::facade::Mnema;
        let mut store = Mnema::new(HashEmbedder::new(HashEmbedder::DEFAULT_DIMS));
        // remember without content, recall without query, remember_fact without value,
        // forget without contains, beliefs without subject: all must FAIL, not silently
        // store/return with "" — success would teach the agent a malformed call worked.
        for call in [
            json!({"name": "remember", "arguments": {}}),
            json!({"name": "remember", "arguments": {"content": 42}}),
            json!({"name": "recall", "arguments": {}}),
            json!({"name": "remember_fact", "arguments": {"subject": "s", "attribute": "a"}}),
            json!({"name": "forget", "arguments": {}}),
            json!({"name": "beliefs", "arguments": {}}),
        ] {
            let r = handle_tool_call(&mut store, "", b"k", Destination::Remote, Some(&call));
            assert_eq!(
                r["isError"],
                json!(true),
                "call {call} must be rejected, got: {r}"
            );
        }
        assert_eq!(store.len(), 0, "no memory may be stored by a rejected call");
    }

    #[test]
    fn recall_k_caps_the_returned_list_not_each_retriever() {
        use mnema::embed::HashEmbedder;
        use mnema::facade::Mnema;
        let mut store = Mnema::new(HashEmbedder::new(HashEmbedder::DEFAULT_DIMS));
        // Plenty of memories sharing tokens with the query, so every fused retriever
        // (lexical/dense/recency) has more than k candidates of its own.
        for i in 0..12 {
            store.remember(EgressTier::Open, &format!("rust memo number {i}"));
        }
        let call = json!({"name": "recall", "arguments": {"query": "rust memo", "k": 2, "budget": 100000}});
        let r = handle_tool_call(&mut store, "", b"k", Destination::Remote, Some(&call));
        let text = r["content"][0]["text"].as_str().unwrap();
        assert_eq!(
            text.lines().count(),
            2,
            "k=2 must return at most 2 memories, got: {text}"
        );
    }

    /// A unique, writable temp directory (removed on drop) for the tool branches that persist —
    /// `path()` is a store path whose parent exists, so `persist` really writes.
    struct TempStore(std::path::PathBuf);

    impl TempStore {
        fn new(label: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("mnema_server_unit_{label}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> String {
            self.0.join("s.store").to_str().unwrap().to_string()
        }
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fresh_store() -> Mnema<mnema::embed::HashEmbedder> {
        use mnema::embed::HashEmbedder;
        Mnema::new(HashEmbedder::new(HashEmbedder::DEFAULT_DIMS))
    }

    /// Call one tool and return its result payload.
    fn call_tool(store: &mut Mnema<mnema::embed::HashEmbedder>, path: &str, call: &Value) -> Value {
        handle_tool_call(store, path, b"k", Destination::Remote, Some(call))
    }

    fn text_of(reply: &Value) -> &str {
        reply["content"][0]["text"].as_str().unwrap()
    }

    #[test]
    fn prompts_list_marks_the_query_argument_required() {
        // The recall prompt is useless without a query; the schema must say so, or a client
        // will happily invoke it with nothing to recall about.
        let p = prompts_list();
        assert_eq!(p["prompts"][0]["arguments"][0]["required"], json!(true));
    }

    #[test]
    fn read_line_capped_returns_a_short_final_unterminated_line() {
        // A last line ending at EOF without `\n` is a complete Line (first operand of the cap
        // check true, second false) — treating it as Oversized would drop a client's final
        // message just because its writer closed the pipe without a trailing newline.
        let mut r: &[u8] = b"{\"jsonrpc\":\"2.0\"}";
        match read_line_capped(&mut r).unwrap() {
            LineRead::Line(b) => assert_eq!(b, b"{\"jsonrpc\":\"2.0\"}"),
            LineRead::Oversized => panic!("a short unterminated final line is not Oversized"),
            LineRead::Eof => panic!("a non-empty read is not Eof"),
        }
    }

    #[test]
    fn blank_lines_are_skipped_without_a_reply() {
        // Blank / whitespace-only lines are keepalive noise, not messages: no reply, not even
        // a parse error — exactly one reply for the one real request.
        let replies = drive(concat!(
            "\n",
            "   \n",
            r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
            "\n"
        ));
        assert_eq!(
            replies.len(),
            1,
            "blank lines must be skipped silently: {replies:?}"
        );
        assert_eq!(replies[0]["id"], json!(1));
    }

    #[test]
    fn recall_on_an_empty_store_reports_no_relevant_memories() {
        let mut store = fresh_store();
        let call = json!({"name": "recall", "arguments": {"query": "anything at all"}});
        let r = call_tool(&mut store, "unused", &call);
        assert_eq!(text_of(&r), "(no relevant memories)");
    }

    #[test]
    fn recent_reports_the_empty_store_and_lists_stored_memories() {
        let mut store = fresh_store();
        let call = json!({"name": "recent", "arguments": {}});
        let r = call_tool(&mut store, "unused", &call);
        assert_eq!(text_of(&r), "(no memories yet)");

        store.remember(EgressTier::Open, "the sky is blue");
        let r = call_tool(&mut store, "unused", &call);
        let text = text_of(&r);
        assert!(text.contains("the sky is blue"), "{text}");
        assert!(!text.contains("(no memories yet)"), "{text}");
    }

    #[test]
    fn remember_fact_confirmation_never_echoes_a_private_value() {
        let ts = TempStore::new("fact_echo");
        let path = ts.path();
        let mut store = fresh_store();
        // A Private belief's storage confirmation must not leak the value back out — the
        // reply goes to the same (possibly remote) model the egress wall protects against.
        let private = json!({"name": "remember_fact", "arguments":
            {"subject": "user", "attribute": "token", "value": "sk-hush", "tier": "private"}});
        let r = call_tool(&mut store, &path, &private);
        assert!(r.get("isError").is_none(), "{r}");
        let text = text_of(&r);
        assert!(text.contains("<private>"), "{text}");
        assert!(
            !text.contains("sk-hush"),
            "a private value must not be echoed: {text}"
        );
        // An Open belief's confirmation echoes the stored value, not the private placeholder.
        let open = json!({"name": "remember_fact", "arguments":
            {"subject": "user", "attribute": "editor", "value": "helix", "tier": "open"}});
        let r = call_tool(&mut store, &path, &open);
        let text = text_of(&r);
        assert!(text.contains("\"helix\""), "{text}");
        assert!(!text.contains("<private>"), "{text}");
    }

    #[test]
    fn beliefs_reports_nothing_known_or_the_live_facts() {
        let mut store = fresh_store();
        let call = json!({"name": "beliefs", "arguments": {"subject": "mars"}});
        let r = call_tool(&mut store, "unused", &call);
        assert_eq!(text_of(&r), "(nothing known about mars)");

        store.remember_fact_tiered("mars", "color", "red", EgressTier::Open);
        let r = call_tool(&mut store, "unused", &call);
        let text = text_of(&r);
        assert!(text.contains("mars.color = red"), "{text}");
        assert!(!text.contains("nothing known"), "{text}");
    }

    #[test]
    fn forget_fact_requires_a_subject_and_deletes_only_the_matching_facts() {
        let ts = TempStore::new("forget_fact");
        let path = ts.path();
        let mut store = fresh_store();
        store.remember_fact_tiered("alice", "color", "red", EgressTier::Open);
        store.remember_fact_tiered("alice", "food", "pizza", EgressTier::Open);
        store.remember_fact_tiered("bob", "color", "blue", EgressTier::Open);

        // No subject: rejected, nothing deleted.
        let bad = json!({"name": "forget_fact", "arguments": {}});
        let r = call_tool(&mut store, &path, &bad);
        assert_eq!(r["isError"], json!(true), "{r}");
        assert_eq!(store.beliefs("alice", Destination::Local).len(), 2);

        // subject + attribute: exactly alice.color goes — alice.food and bob.color both stay.
        let call =
            json!({"name": "forget_fact", "arguments": {"subject": "alice", "attribute": "color"}});
        let r = call_tool(&mut store, &path, &call);
        let text = text_of(&r);
        assert!(
            text.contains("forgot 1 belief record(s) for alice.color"),
            "{text}"
        );
        let alice: Vec<String> = store
            .beliefs("alice", Destination::Local)
            .iter()
            .map(|f| format!("{}.{}", f.subject, f.attribute))
            .collect();
        assert!(alice.contains(&"alice.food".to_string()), "{alice:?}");
        assert!(!alice.contains(&"alice.color".to_string()), "{alice:?}");
        assert_eq!(
            store.beliefs("bob", Destination::Local).len(),
            1,
            "bob must be untouched"
        );

        // subject only: everything about alice goes — bob still stays.
        let call = json!({"name": "forget_fact", "arguments": {"subject": "alice"}});
        let r = call_tool(&mut store, &path, &call);
        let text = text_of(&r);
        assert!(
            text.contains("forgot 1 belief record(s) for alice"),
            "{text}"
        );
        assert!(store.beliefs("alice", Destination::Local).is_empty());
        assert_eq!(store.beliefs("bob", Destination::Local).len(), 1);
    }

    #[test]
    fn reinforce_reports_the_found_and_not_found_sides() {
        let ts = TempStore::new("reinforce");
        let path = ts.path();
        let mut store = fresh_store();
        let id = store.remember(EgressTier::Open, "worth keeping");

        let call = json!({"name": "reinforce", "arguments": {"id": id}});
        let r = call_tool(&mut store, &path, &call);
        assert_eq!(text_of(&r), format!("reinforced memory {id}"));

        let call = json!({"name": "reinforce", "arguments": {"id": 999_999}});
        let r = call_tool(&mut store, &path, &call);
        assert_eq!(text_of(&r), "no memory with id 999999");
    }

    #[test]
    fn forget_deletes_exactly_the_matching_memories() {
        let ts = TempStore::new("forget");
        let path = ts.path();
        let mut store = fresh_store();
        store.remember(EgressTier::Open, "apple pie recipe");
        store.remember(EgressTier::Open, "banana bread");

        // An empty needle matches nothing — `forget ""` must not wipe the store.
        let call = json!({"name": "forget", "arguments": {"contains": ""}});
        let r = call_tool(&mut store, &path, &call);
        assert_eq!(text_of(&r), "forgot 0 memories; 2 remain");

        // A real needle deletes exactly the memories containing it.
        let call = json!({"name": "forget", "arguments": {"contains": "apple"}});
        let r = call_tool(&mut store, &path, &call);
        assert_eq!(text_of(&r), "forgot 1 memories; 1 remain");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn lock_store_preserves_existing_lock_file_content() {
        // The lock file is opened with truncate(false): pre-existing content must survive
        // taking and releasing the lock.
        let ts = TempStore::new("locktrunc");
        let path = ts.path();
        let lockpath = format!("{path}.lock");
        std::fs::write(&lockpath, b"sentinel").unwrap();
        let file = lock_store(&path);
        drop(file); // release before reading: an exclusive lock can block reads on Windows
        assert_eq!(
            std::fs::read(&lockpath).unwrap(),
            b"sentinel",
            "taking the store lock must not truncate the lock file"
        );
    }

    #[test]
    fn ensure_parent_dir_creates_a_missing_store_directory() {
        let mut base = std::env::temp_dir();
        base.push("mnema_ensure_parent_dir_test");
        let _ = std::fs::remove_dir_all(&base);
        let nested = base.join("nested").join("deeper");
        let store = nested.join("s.store");
        assert!(!nested.exists());
        ensure_parent_dir(store.to_str().unwrap());
        assert!(
            nested.is_dir(),
            "the store's parent directory must be created"
        );
        ensure_parent_dir(store.to_str().unwrap()); // idempotent on an existing dir
        let _ = std::fs::remove_dir_all(&base);
    }
}
