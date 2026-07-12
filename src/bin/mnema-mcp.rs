//! `mnema-mcp` — a Model Context Protocol server exposing mnema's memory as tools over stdio.
//!
//! Any MCP client (Claude Code, Cursor, Claude Desktop) can give an agent **private, local,
//! encrypted** memory by pointing it at this binary. Configure the store with two env vars:
//!
//! ```jsonc
//! // e.g. in an MCP client config
//! { "command": "mnema-mcp", "env": { "MNEMA_PATH": "~/mnema.store", "MNEMA_KEY": "your-passphrase" } }
//! ```
//!
//! This is **below-waterline glue**: it speaks line-delimited JSON-RPC and persists the store,
//! but every real decision — the egress wall, contradiction resolution, packing — lives in the
//! mutation-pinned [`mnema`] facade. Notably, `recall` runs against `Destination::Remote`, so a
//! `Private` memory can never leave through this tool.
//!
//! ## Embedder
//!
//! By default recall rides the zero-dependency [`HashEmbedder`] (a lexical bag-of-tokens vector).
//! Built with `--features mcp,local-embed`, it instead loads the in-process semantic embedder
//! (`all-MiniLM-L6-v2` via candle), so recall matches on *meaning*, not just shared words. The
//! choice is fixed at compile time; a store is embedder-specific (the widths differ), so a given
//! binary always opens the store it wrote.

use std::io::{BufRead, Write};

#[cfg(not(feature = "local-embed"))]
use mnema::embed::HashEmbedder;
use mnema::facade::Mnema;
use mnema::vector::Embedder;
use mnema::{BundleItem, Destination, EgressTier};
use serde_json::{Value, json};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let path = std::env::var("MNEMA_PATH").unwrap_or_else(|_| "mnema.store".to_string());
    // Hold an exclusive advisory lock on the store for the whole session. A resident server keeps
    // the store in RAM and re-seals it on every write, so a second concurrent writer's records
    // would be silently clobbered by the next re-seal. `_lock` lives to the end of main; the OS
    // releases it when the process exits (even on crash). Taken first, so a keyfile generated below
    // (for a fresh store) is written under the lock.
    let _lock = lock_store(&path);

    // Resolve the per-store key the same way the CLI does: $MNEMA_KEY, else a random sidecar
    // <store>.key (generated for a fresh store). This shares a store family with the CLI and never
    // seals under an empty passphrase — an unset key no longer means weak, silent encryption.
    let key = mnema::keyfile::resolve_key(std::path::Path::new(&path)).unwrap_or_else(|e| {
        eprintln!("mnema-mcp: {e}");
        std::process::exit(1);
    });

    // The embedder is chosen at compile time. `serve` is generic over it, so the whole server
    // is identical either way — only the vector arm of recall differs (lexical vs semantic).
    #[cfg(feature = "local-embed")]
    let store = open_store(&path, &key, || {
        mnema::model_embed::MiniLmEmbedder::load().unwrap_or_else(|e| {
            eprintln!("mnema-mcp: could not load the semantic model ({e})");
            std::process::exit(1);
        })
    });
    #[cfg(not(feature = "local-embed"))]
    let store = open_store(&path, &key, || {
        HashEmbedder::new(HashEmbedder::DEFAULT_DIMS)
    });

    serve(store, &path, &key);
}

/// The JSON-RPC read/dispatch/persist loop, generic over the embedder in play.
fn serve<E: Embedder>(mut store: Mnema<E>, path: &str, key: &[u8]) {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            // A per-line read error (e.g. a non-UTF-8 byte on stdin) is NOT end-of-input: report
            // it and keep serving, so one bad byte can't silently kill the whole session. Only a
            // genuine broken pipe / closed stdin ends the loop.
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                write_msg(
                    &mut stdout,
                    &error(Value::Null, -32700, "parse error: non-UTF-8 input"),
                );
                continue;
            }
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(line) else {
            write_msg(&mut stdout, &error(Value::Null, -32700, "parse error"));
            continue;
        };
        let id = req.get("id").cloned();
        let method = req
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match method {
            "initialize" => write_msg(&mut stdout, &reply(id, initialize_result())),
            "tools/list" => write_msg(&mut stdout, &reply(id, tools_list())),
            "tools/call" => {
                let r = handle_tool_call(&mut store, path, key, req.get("params"));
                write_msg(&mut stdout, &reply(id, r));
            }
            "ping" => write_msg(&mut stdout, &reply(id, json!({}))),
            // Notifications carry no id and expect no response.
            _ if method.starts_with("notifications/") => {}
            _ => {
                if let Some(id) = id {
                    write_msg(&mut stdout, &error(id, -32601, "method not found"));
                }
            }
        }
    }
}

fn write_msg(out: &mut impl Write, msg: &Value) {
    let _ = writeln!(out, "{msg}");
    let _ = out.flush();
}

fn reply(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result })
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "mnema", "version": SERVER_VERSION }
    })
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

    let text = match name {
        "remember" => {
            let content = arg_str("content");
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
            let query = arg_str("query");
            let k = args.get("k").and_then(Value::as_u64).unwrap_or(5) as usize;
            let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2000) as usize;
            // half_life 0 keeps importance weighting (a memory marked important ranks higher)
            // without time decay; a positive value also biases toward recent memories.
            // Destination::Remote drops Private memories at the egress wall either way.
            let half_life = args.get("half_life").and_then(Value::as_u64).unwrap_or(0);
            let hits = store.recall_decayed(&query, Destination::Remote, k, budget, half_life);
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
            let mut items = store.recall_recent(Destination::Remote, budget);
            items.truncate(k);
            if items.is_empty() {
                "(no memories yet)".to_string()
            } else {
                render_items(&items)
            }
        }
        "remember_fact" => {
            let (s, a, v) = (arg_str("subject"), arg_str("attribute"), arg_str("value"));
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
            let subject = arg_str("subject");
            // Destination::Remote applies the egress wall: Private beliefs are withheld.
            let facts = store.beliefs(&subject, Destination::Remote);
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
            let needle = arg_str("contains");
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
    std::fs::rename(&tmp, path)
}

/// Take an exclusive advisory lock for the store via a sibling `<path>.lock`, returning the held
/// `File` (drop it to release; the OS also releases on process exit). Refuses to start if another
/// mnema process already holds it, so two writers can't clobber each other. The lock file is
/// separate from the store because the store is atomically *renamed* on every write, which would
/// drop a lock held on the store file itself.
fn lock_store(path: &str) -> std::fs::File {
    let lockpath = format!("{path}.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lockpath)
        .unwrap_or_else(|e| {
            eprintln!("mnema-mcp: cannot open lock file {lockpath} ({e})");
            std::process::exit(1);
        });
    if file.try_lock().is_err() {
        eprintln!(
            "mnema-mcp: {path} is already in use by another mnema process — refusing to start so a \
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
            Err(e) => {
                eprintln!(
                    "mnema-mcp: {path} exists but could not be opened ({e:?}) — wrong MNEMA_KEY \
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
                "mnema-mcp: could not read {path} ({e}) — refusing to start so an existing store \
                 is not overwritten by an empty one. Resolve the I/O error (or move the file) and retry."
            );
            std::process::exit(1);
        }
    }
}
