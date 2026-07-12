//! `mnema` — a tiny CLI over the memory layer, so non-Rust callers can remember and recall
//! across runs.
//!
//! Deliberately thin: every real decision lives in the mutation-pinned facade; this only
//! parses args, loads/seals the on-disk store, and prints. It is I/O orchestration below the
//! mutation gate's behavioral waterline, so it is not part of the probed `sources`. The store
//! is one sealed blob (ADR-0020 crypto).
//!
//! The key is per-store, resolved in this order (never on the command line):
//!   1. `$MNEMA_KEY` if set — an explicit passphrase (shared stores, CI, env-only secrets);
//!   2. else a random 32-byte key in the sidecar `<store>.key`, generated on first use.
//!
//! There is no shared default: each store gets its own independent key. To migrate a store
//! that was sealed under an old passphrase, `mnema rekey <store>` (with `$MNEMA_KEY` set
//! to the old passphrase) re-seals it under a fresh keyfile.
//!
//! Usage:
//!   mnema remember <store> <open|redacted|private> <content>   # prints the new id
//!   mnema fact     <store> <subject> <attribute> <value>       # prints the resolution
//!   mnema recall   <store> <k> <query>                         # prints k memories
//!   mnema stats    <store>
//!   mnema prune    <store> <half_life> <threshold>             # forget faded memories
//!   mnema rekey    <store>   # $MNEMA_KEY = old passphrase; re-seals under a new keyfile

use std::io::Write;
use std::path::Path;
use std::process::exit;

use mnema::embed::HashEmbedder;
use mnema::facade::Mnema;
use mnema::keyfile::{self, generate_keyfile, keyfile_path};
use mnema::{Destination, EgressTier};

/// The default embedder's width, pinned once in the library so this CLI and the
/// `mnema-mcp` server — which share a store family — can never embed at different
/// widths and corrupt each other's recall.
const DIMS: usize = HashEmbedder::DEFAULT_DIMS;

fn die(msg: &str) -> ! {
    eprintln!("mnema: {msg}");
    exit(1);
}

/// The per-store key via the shared [`keyfile::resolve_key`], mapping any failure to a CLI exit
/// (the migration case gets the `rekey` hint).
fn resolve_key(store: &Path) -> Vec<u8> {
    keyfile::resolve_key(store).unwrap_or_else(|e| die(&e.to_string()))
}

fn load(store: &str) -> Mnema<HashEmbedder> {
    let embedder = HashEmbedder::new(DIMS);
    let path = Path::new(store);
    if path.exists() {
        let bytes = std::fs::read(store).unwrap_or_else(|e| die(&format!("read {store}: {e}")));
        Mnema::open(&bytes, &resolve_key(path), embedder).unwrap_or_else(|_| {
            die(
                "cannot open store (wrong key or corrupt). If a `rekey` was interrupted, set \
                 $MNEMA_KEY to the OLD passphrase and re-run `mnema rekey <store>` to finish it.",
            )
        })
    } else {
        Mnema::new(embedder)
    }
}

fn save(store: &str, mem: &mut Mnema<HashEmbedder>) {
    let blob = mem
        .seal(&resolve_key(Path::new(store)))
        .unwrap_or_else(|_| die("seal failed"));
    write_atomic(store, &blob).unwrap_or_else(|e| die(&format!("write {store}: {e}")));
}

/// Write `bytes` to `path` durably: write a sibling `.tmp`, flush it to disk, then rename it
/// over `path`. The rename is atomic within the directory, so a crash or full disk mid-write
/// can never leave `path` a torn blob — it stays either the whole old store or the whole new
/// one, and the original is untouched on any failure.
/// Take an exclusive advisory lock for the store via a sibling `<store>.lock`, returning the held
/// `File` (drop to release; the OS releases on exit). A write command holds this across its
/// load→mutate→save so it can't clobber, or be clobbered by, a concurrent writer (another `mnema`
/// or a running `mnema-mcp`). Read-only commands don't lock — writes are atomic, so a reader sees
/// the whole old or whole new store, never a torn one.
fn lock_store(store: &str) -> std::fs::File {
    let lockpath = format!("{store}.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lockpath)
        .unwrap_or_else(|e| die(&format!("cannot open lock file {lockpath}: {e}")));
    if file.try_lock().is_err() {
        die(&format!(
            "{store} is in use by another mnema process; retry once it exits"
        ));
    }
    file
}

fn write_atomic(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = format!("{path}.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// Migrate a store to a per-store keyfile: open it with the current `$MNEMA_KEY`, then re-seal it
/// under a `<store>.key`.
///
/// **Crash-safe by ordering + resume.** The keyfile is written *before* the store is sealed under
/// it, so the new key is always durable before any blob depends on it (sealing store-first could
/// commit a store under a random key that is then lost if the keyfile write fails — an
/// unrecoverable store). A crash in the window leaves keyfile=new, store=still-under-old — the
/// store is intact and openable with the old passphrase. Re-running `rekey` (with `$MNEMA_KEY` =
/// the old passphrase) then **resumes**: it reuses the existing keyfile and finishes the re-seal,
/// rather than refusing. A store already fully migrated will not open under the old passphrase, so
/// it cannot be clobbered.
fn rekey(store: &str) {
    let path = Path::new(store);
    if !path.exists() {
        die(&format!("rekey: store {store} does not exist"));
    }
    let _lock = lock_store(store);
    let old = match std::env::var("MNEMA_KEY") {
        Ok(k) if !k.is_empty() => k.into_bytes(),
        _ => die("rekey: set $MNEMA_KEY to the store's CURRENT passphrase"),
    };
    let bytes = std::fs::read(store).unwrap_or_else(|e| die(&format!("read {store}: {e}")));
    let mut mem = Mnema::open(&bytes, &old, HashEmbedder::new(DIMS))
        .unwrap_or_else(|_| die("rekey: cannot open store with $MNEMA_KEY (wrong passphrase?)"));

    // Reuse an existing keyfile to resume an interrupted rekey; otherwise generate + persist one.
    // Either way the key is on disk before we seal the store under it.
    let keyfile = keyfile_path(path);
    let new_key = match std::fs::read(&keyfile) {
        Ok(k) if k.len() == 32 => k,
        Ok(_) => die(&format!(
            "rekey: {} is malformed (expected 32 bytes); remove it and retry",
            keyfile.display()
        )),
        Err(_) => generate_keyfile(&keyfile).unwrap_or_else(|e| die(&format!("rekey: {e}"))),
    };
    let blob = mem
        .seal(&new_key)
        .unwrap_or_else(|_| die("rekey: seal failed"));
    write_atomic(store, &blob).unwrap_or_else(|e| die(&format!("write {store}: {e}")));
    println!("rekeyed {store} under {}", keyfile.display());
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
            let _lock = lock_store(store);
            let mut mem = load(store);
            let id = mem.remember(tier(&args[2]), &args[3]);
            save(store, &mut mem);
            println!("{id}");
        }
        ("fact", 5) => {
            let store = &args[1];
            let _lock = lock_store(store);
            let mut mem = load(store);
            let res = mem.remember_fact(&args[2], &args[3], &args[4]);
            save(store, &mut mem);
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
            let s = load(&args[1]).stats();
            println!(
                "memories: {} (open {}, redacted {}, private {})",
                s.total, s.open, s.redacted, s.private
            );
            println!(
                "beliefs: {} live ({} private)",
                s.beliefs, s.private_beliefs
            );
            println!("indexed: {}", s.indexed);
        }
        ("prune", 4) => {
            let store = &args[1];
            let _lock = lock_store(store);
            let half_life: u64 = args[2]
                .parse()
                .unwrap_or_else(|_| die("half_life must be a number"));
            let threshold: f32 = args[3]
                .parse()
                .unwrap_or_else(|_| die("threshold must be a number"));
            let mut mem = load(store);
            let receipt = mem.prune_faded(half_life, threshold);
            save(store, &mut mem);
            println!(
                "pruned {}  remaining {}",
                receipt.purged.len(),
                receipt.remaining
            );
        }
        ("rekey", 2) => rekey(&args[1]),
        _ => die(
            "usage: mnema remember|recall|fact|stats|prune|rekey <store> ...  (see the source header)",
        ),
    }
}
