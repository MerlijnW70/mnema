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

use std::io::{BufRead, Write};

use mnema::embed::HashEmbedder;
use mnema::facade::Mnema;
use mnema::{Destination, EgressTier};
use serde_json::{Value, json};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let path = std::env::var("MNEMA_PATH").unwrap_or_else(|_| "mnema.store".to_string());
    let key = std::env::var("MNEMA_KEY").unwrap_or_default();
    if key.is_empty() {
        eprintln!("mnema-mcp: MNEMA_KEY is empty — set a passphrase to encrypt the store at rest");
    }

    let mut store = load_store(&path, key.as_bytes());
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
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
                let r = handle_tool_call(&mut store, &path, key.as_bytes(), req.get("params"));
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
            "description": "Store a memory. 'tier' controls egress: a 'private' memory is never returned to a remote/cloud model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "the memory text to store" },
                    "tier": { "type": "string", "enum": ["open", "redacted", "private"], "description": "egress tier (default: open)" }
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
                    "budget": { "type": "integer", "description": "max total characters (default 2000)" }
                },
                "required": ["query"]
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
            "description": "Report how many memories are stored and indexed.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ]})
}

fn tool_text(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

fn handle_tool_call(
    store: &mut Mnema<HashEmbedder>,
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
            let tier = parse_tier(args.get("tier").and_then(Value::as_str));
            let id = store.remember(tier, &content);
            persist(store, path, key);
            format!("remembered as memory {id}")
        }
        "recall" => {
            let query = arg_str("query");
            let k = args.get("k").and_then(Value::as_u64).unwrap_or(5) as usize;
            let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2000) as usize;
            // Destination::Remote applies the egress wall: Private memories are dropped here.
            let hits = store.recall(&query, Destination::Remote, k, budget);
            if hits.is_empty() {
                "(no relevant memories)".to_string()
            } else {
                hits.iter()
                    .map(|b| format!("[{}] {}", b.id, b.text))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        "remember_fact" => {
            let (s, a, v) = (arg_str("subject"), arg_str("attribute"), arg_str("value"));
            let tier = parse_tier(args.get("tier").and_then(Value::as_str));
            let res = store.remember_fact_tiered(&s, &a, &v, tier);
            persist(store, path, key);
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
        "forget" => {
            let needle = arg_str("contains");
            let receipt = store.forget(|m| !needle.is_empty() && m.content.contains(&needle));
            persist(store, path, key);
            format!(
                "forgot {} memories; {} remain",
                receipt.purged.len(),
                receipt.remaining
            )
        }
        "stats" => format!("{} memories, {} indexed", store.len(), store.indexed()),
        other => {
            return json!({
                "content": [{ "type": "text", "text": format!("unknown tool: {other}") }],
                "isError": true
            });
        }
    };
    tool_text(text)
}

fn parse_tier(s: Option<&str>) -> EgressTier {
    match s {
        Some("private") => EgressTier::Private,
        Some("redacted") => EgressTier::Redacted,
        _ => EgressTier::Open,
    }
}

fn persist(store: &mut Mnema<HashEmbedder>, path: &str, key: &[u8]) {
    match store.seal(key) {
        Ok(blob) => {
            if let Err(e) = std::fs::write(path, blob) {
                eprintln!("mnema-mcp: failed to write store to {path}: {e}");
            }
        }
        Err(e) => eprintln!("mnema-mcp: failed to seal store: {e:?}"),
    }
}

fn load_store(path: &str, key: &[u8]) -> Mnema<HashEmbedder> {
    let dims = HashEmbedder::DEFAULT_DIMS;
    match std::fs::read(path) {
        Ok(blob) => match Mnema::open(&blob, key, HashEmbedder::new(dims)) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("mnema-mcp: could not open {path} ({e:?}); starting a fresh store");
                Mnema::new(HashEmbedder::new(dims))
            }
        },
        Err(_) => Mnema::new(HashEmbedder::new(dims)),
    }
}
