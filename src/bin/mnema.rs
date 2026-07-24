// A binary is its own crate root, so the library's `#![forbid(unsafe_code)]` does not reach
// here. Without this line these 400-odd lines would be the one part of the workspace where
// `unsafe` could appear unnoticed.
#![forbid(unsafe_code)]

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
//!   mnema remember    <store> <open|redacted|private> <content>   # prints the new id
//!   mnema fact        <store> <subject> <attribute> <value>       # prints the resolution
//!   mnema recall      <store> <k> <query>                         # prints k memories
//!   mnema recent      <store> <k>                                 # k most recent memories
//!   mnema beliefs     <store> <subject>                           # live beliefs about subject
//!   mnema reinforce   <store> <id>                                # strengthen a memory
//!   mnema forget      <store> <substring>                         # hard-delete matching memories
//!   mnema forget-fact <store> <subject> [attribute]               # hard-delete beliefs
//!   mnema stats       <store>
//!   mnema prune       <store> <half_life> <threshold>             # forget faded memories
//!   mnema rekey       <store>   # $MNEMA_KEY = old passphrase; re-seals under a new keyfile
//!   mnema keygen                # print a strong random passphrase to use as $MNEMA_KEY

use std::io::Write;
use std::path::Path;
use std::process::exit;

use mnema::embed::HashEmbedder;
use mnema::facade::Mnema;
use mnema::keyfile::{self, generate_keyfile, keyfile_path};
use mnema::{Destination, EgressTier};

/// The default embedder's width, pinned once in the library so this CLI and the
/// `mnema-server` server — which share a store family — can never embed at different
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
    try_load(store).unwrap_or_else(|msg| die(&msg))
}

/// Open the store at `store`, or start fresh **only if there is genuinely no file yet**.
///
/// Fail-closed on both axes a memory product cannot get wrong:
/// - ONLY `NotFound` means "start fresh". Any other read error (permissions, a sharing
///   violation, transient I/O) must NOT begin empty — the next `save` would overwrite, and
///   destroy, the real store we merely failed to read. (`Path::exists()` cannot make this
///   distinction: it reads `false` on a stat *error*, which is exactly the overwrite trap.)
/// - The open error is *reported by cause*: a vector-width mismatch means "different
///   embedder" (fix: migrate), an unknown format version means "newer mnema" (fix: upgrade)
///   — telling the user "wrong key" for those would send them to the wrong repair.
fn try_load(store: &str) -> Result<Mnema<HashEmbedder>, String> {
    match std::fs::read(store) {
        Ok(bytes) => {
            let key = resolve_key(Path::new(store));
            try_open(store, &bytes, &key)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok(Mnema::new(HashEmbedder::new(DIMS)))
        }
        Err(e) => Err(format!(
            "read {store}: {e} — refusing to start an empty store over one that may exist. \
             Resolve the I/O error and retry."
        )),
    }
}

/// Open an already-read store blob under `key`, mapping each failure to the repair the user
/// actually needs (see [`try_load`]).
fn try_open(store: &str, bytes: &[u8], key: &[u8]) -> Result<Mnema<HashEmbedder>, String> {
    use mnema::store::StoreError;
    match Mnema::open(bytes, key, HashEmbedder::new(DIMS)) {
        Ok(m) => Ok(m),
        Err(StoreError::EmbedderWidthMismatch { stored, embedder }) => Err(format!(
            "{store} was written by a different embedder (vector width {stored}, this build \
             uses {embedder}). Re-embed it once with `mnema-server --migrate --path {store}`, \
             then retry."
        )),
        Err(StoreError::UnknownVersion) => Err(format!(
            "{store} uses an on-disk format newer than this mnema understands — upgrade mnema."
        )),
        Err(e) => Err(format!(
            "cannot open {store} ({e:?}) — wrong key or corrupt. If a `rekey` was \
             interrupted, set $MNEMA_KEY to the OLD passphrase and re-run \
             `mnema rekey {store}` to finish it."
        )),
    }
}

fn save(store: &str, mem: &mut Mnema<HashEmbedder>) {
    let blob = mem
        .seal(&resolve_key(Path::new(store)))
        .unwrap_or_else(|e| die(&format!("seal failed ({e:?})")));
    write_atomic(store, &blob).unwrap_or_else(|e| die(&format!("write {store}: {e}")));
}

/// Take an exclusive advisory lock for the store via a sibling `<store>.lock`, returning the held
/// `File` (drop to release; the OS releases on exit). A write command holds this across its
/// load→mutate→save so it can't clobber, or be clobbered by, a concurrent writer (another `mnema`
/// or a running `mnema-server`). Read-only commands don't lock — writes are atomic, so a reader sees
/// the whole old or whole new store, never a torn one.
fn lock_store(store: &str) -> std::fs::File {
    // Create the store's directory if it doesn't exist yet, so a store path into a not-yet-created
    // folder just works rather than failing with a cryptic "cannot open lock file … path not found".
    if let Some(dir) = std::path::Path::new(store)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        die(&format!(
            "cannot create store directory {}: {e}",
            dir.display()
        ));
    }
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

/// Write `bytes` to `path` durably: write a sibling `.tmp`, flush it to disk, then rename it
/// over `path`. The rename is atomic within the directory, so a crash or full disk mid-write
/// can never leave `path` a torn blob — it stays either the whole old store or the whole new
/// one, and the original is untouched on any failure.
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
    Ok(())
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

/// The `forget-fact` deletion predicate: a belief is removed when its subject matches `subject`
/// AND (no attribute filter was given, or its attribute matches). Split out of the inline
/// closure so it is unit-testable in-process — the exact `subject == …` / `&&` / `||` boundary
/// is load-bearing (get it wrong and `forget-fact` deletes the wrong beliefs), so it earns a
/// direct test rather than only end-to-end coverage through the spawned binary.
fn forget_fact_matches(subject: &str, attribute: &str, f_subject: &str, f_attribute: &str) -> bool {
    f_subject == subject && (attribute.is_empty() || f_attribute == attribute)
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
        ("recent", 3) => {
            let store = &args[1];
            let k: usize = args[2]
                .parse()
                .unwrap_or_else(|_| die("k must be a number"));
            // A local tool: Destination::Local, so Private memories are visible to their owner.
            let mut items = load(store).recall_recent(Destination::Local, 100_000);
            items.truncate(k);
            for item in items {
                println!("[{}] {}", item.id, item.text);
            }
        }
        ("beliefs", 3) => {
            let store = &args[1];
            for f in load(store).beliefs(&args[2], Destination::Local) {
                println!("{}.{} = {}", f.subject, f.attribute, f.value);
            }
        }
        ("reinforce", 3) => {
            let store = &args[1];
            let id: u64 = args[2]
                .parse()
                .unwrap_or_else(|_| die("id must be a number"));
            let _lock = lock_store(store);
            let mut mem = load(store);
            if mem.reinforce(id) {
                save(store, &mut mem);
                println!("reinforced {id}");
            } else {
                die(&format!("no memory with id {id}"));
            }
        }
        ("forget", 3) => {
            let store = &args[1];
            let needle = &args[2];
            let _lock = lock_store(store);
            let mut mem = load(store);
            let receipt = mem.forget(|m| m.content.contains(needle.as_str()));
            save(store, &mut mem);
            println!(
                "forgot {}  remaining {}",
                receipt.purged.len(),
                receipt.remaining
            );
        }
        ("forget-fact", 3) | ("forget-fact", 4) => {
            let store = &args[1];
            let subject = &args[2];
            let attribute = args.get(3).cloned().unwrap_or_default();
            let _lock = lock_store(store);
            let mut mem = load(store);
            let removed = mem.forget_facts(|f| {
                forget_fact_matches(subject, &attribute, &f.subject, &f.attribute)
            });
            save(store, &mut mem);
            println!("forgot {removed} belief record(s)");
        }
        ("rekey", 2) => rekey(&args[1]),
        ("keygen", 1) => {
            // A strong random passphrase to set as $MNEMA_KEY (e.g. `export MNEMA_KEY=$(mnema keygen)`).
            // For a portable secret; if you don't need one, just omit $MNEMA_KEY and a per-store
            // keyfile is generated for you.
            println!(
                "{}",
                keyfile::generate_passphrase().unwrap_or_else(|e| die(&e.to_string()))
            );
        }
        _ => die(
            "usage: mnema remember|recall|recent|fact|beliefs|reinforce|forget|forget-fact|stats|prune|rekey <store> ... | keygen  (see the source header)",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("mnema_cli_test_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn forget_fact_matches_pins_the_subject_and_attribute_boundary() {
        // Subject must match (== not !=): a different subject never matches, even on a shared
        // attribute — the guard against `forget-fact alice color` deleting bob.color.
        assert!(forget_fact_matches("alice", "color", "alice", "color"));
        assert!(!forget_fact_matches("alice", "color", "bob", "color"));

        // With NO attribute filter (empty), every belief of the subject matches (the `||` short-
        // circuits true) — and none of another subject's.
        assert!(forget_fact_matches("bob", "", "bob", "color"));
        assert!(forget_fact_matches("bob", "", "bob", "size"));
        assert!(!forget_fact_matches("bob", "", "alice", "color"));

        // With an attribute filter, BOTH must match (the `&&`): right subject, wrong attribute
        // does NOT match — distinguishes `&&` from `||`.
        assert!(!forget_fact_matches("alice", "color", "alice", "size"));
    }

    #[test]
    fn try_load_starts_fresh_only_for_a_genuinely_missing_file() {
        let d = temp_dir("fresh");
        let missing = d.join("no-such.store");
        let m = try_load(missing.to_str().unwrap()).expect("a missing file starts fresh");
        assert_eq!(m.len(), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn try_load_refuses_when_the_path_stats_but_cannot_be_read_as_a_store_file() {
        // The overwrite trap this exists to prevent: `Path::exists()`-style logic reads any
        // stat/read error as "no store" and begins empty — then save() destroys the real store.
        // A directory at the store path is a portable stand-in for "read fails, NOT NotFound":
        // the load must REFUSE, never hand back an empty store.
        let d = temp_dir("refuse");
        let r = try_load(d.to_str().unwrap());
        assert!(
            r.is_err(),
            "an unreadable-but-present path must refuse, not begin an empty store"
        );
        assert!(r.err().unwrap().contains("refusing"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn try_open_reports_each_failure_by_its_actual_cause() {
        let key = b"correct horse battery staple";
        // A real store sealed under `key`, holding one memory.
        let mut mem = Mnema::new(HashEmbedder::new(DIMS));
        mem.remember(EgressTier::Open, "the one memory");
        let blob = mem.seal(key).unwrap();

        // Round-trips under the right key.
        let opened = try_open("s", &blob, key).expect("the right key opens the store");
        assert_eq!(opened.len(), 1);

        // A wrong key is reported as the key/corruption case, with the rekey-resume hint.
        let wrong = try_open("s", &blob, b"wrong key").err().unwrap();
        assert!(wrong.contains("wrong key or corrupt"), "{wrong}");

        // A store sealed by a different-width embedder is a MIGRATION case, not "wrong key" —
        // misreporting it would send the user to rekey, which cannot fix it.
        let mut other = Mnema::new(HashEmbedder::new(64));
        other.remember(EgressTier::Open, "written at width 64");
        let other_blob = other.seal(key).unwrap();
        let width = try_open("s", &other_blob, key).err().unwrap();
        assert!(width.contains("different embedder"), "{width}");
        assert!(width.contains("--migrate"), "{width}");

        // An unknown format-version byte is an UPGRADE case, again not "wrong key".
        let mut newer = blob.clone();
        newer[0] = 0xFE;
        let ver = try_open("s", &newer, key).err().unwrap();
        assert!(ver.contains("newer than this mnema"), "{ver}");
    }
}
