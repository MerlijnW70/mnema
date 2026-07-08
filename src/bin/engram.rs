//! `engram` — a tiny CLI over the memory layer, so non-Rust callers (notably the
//! evolution loop, `scripts/evolve.sh`) can remember and recall across runs.
//!
//! Deliberately thin: every real decision lives in the ratchet-pinned facade; this only
//! parses args, loads/seals the on-disk store, and prints. It is I/O orchestration
//! *below noha's behavioral waterline* (Part 23) — like `evolve.sh` itself — so it is
//! not part of the probed `sources`. The store is one sealed blob (ADR-0020 crypto);
//! the passphrase comes from `$ENGRAM_KEY`, never the command line.
//!
//! Usage:
//!   engram remember <store> <open|redacted|private> <content>   # prints the new id
//!   engram fact     <store> <subject> <attribute> <value>       # prints the resolution
//!   engram recall   <store> <k> <query>                         # prints k memories
//!   engram stats    <store>

use std::path::Path;
use std::process::exit;

use engram::embed::HashEmbedder;
use engram::facade::Engram;
use engram::{Destination, EgressTier};

const DIMS: usize = 64;

fn die(msg: &str) -> ! {
    eprintln!("engram: {msg}");
    exit(1);
}

fn key() -> Vec<u8> {
    match std::env::var("ENGRAM_KEY") {
        Ok(k) if !k.is_empty() => k.into_bytes(),
        _ => die("set $ENGRAM_KEY to the store passphrase"),
    }
}

fn load(store: &str) -> Engram<HashEmbedder> {
    let embedder = HashEmbedder::new(DIMS);
    if Path::new(store).exists() {
        let bytes = std::fs::read(store).unwrap_or_else(|e| die(&format!("read {store}: {e}")));
        Engram::open(&bytes, &key(), embedder)
            .unwrap_or_else(|_| die("cannot open store (wrong key or corrupt)"))
    } else {
        Engram::new(embedder)
    }
}

fn save(store: &str, mem: &Engram<HashEmbedder>) {
    let blob = mem.seal(&key()).unwrap_or_else(|_| die("seal failed"));
    std::fs::write(store, blob).unwrap_or_else(|e| die(&format!("write {store}: {e}")));
}

fn tier(s: &str) -> EgressTier {
    match s {
        "open" => EgressTier::Open,
        "redacted" => EgressTier::Redacted,
        "private" => EgressTier::Private,
        _ => die("tier must be open|redacted|private"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    match (cmd, args.len()) {
        ("remember", 4) => {
            let store = &args[1];
            let mut mem = load(store);
            let id = mem.remember(tier(&args[2]), &args[3]);
            save(store, &mem);
            println!("{id}");
        }
        ("fact", 5) => {
            let store = &args[1];
            let mut mem = load(store);
            let res = mem.remember_fact(&args[2], &args[3], &args[4]);
            save(store, &mem);
            println!("{res:?}");
        }
        ("recall", 4) => {
            let store = &args[1];
            let k: usize = args[2]
                .parse()
                .unwrap_or_else(|_| die("k must be a number"));
            let mem = load(store);
            for item in mem.recall(&args[3], Destination::Local, k, 100_000) {
                println!("- {}", item.text);
            }
        }
        ("stats", 2) => {
            let mem = load(&args[1]);
            println!("memories: {}  indexed: {}", mem.len(), mem.indexed());
        }
        _ => die("usage: engram remember|recall|fact|stats <store> ...  (see the source header)"),
    }
}
