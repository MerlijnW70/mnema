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
//!   mnema rekey    <store>   # $MNEMA_KEY = old passphrase; re-seals under a new keyfile

use std::path::{Path, PathBuf};
use std::process::exit;

use mnema::embed::HashEmbedder;
use mnema::facade::Mnema;
use mnema::{Destination, EgressTier};

/// The default embedder's width, pinned once in the library so this CLI and the
/// `mnema-mcp` server — which share a store family — can never embed at different
/// widths and corrupt each other's recall.
const DIMS: usize = HashEmbedder::DEFAULT_DIMS;

fn die(msg: &str) -> ! {
    eprintln!("mnema: {msg}");
    exit(1);
}

/// The sidecar keyfile for a store: `<store>.key`.
fn keyfile_path(store: &Path) -> PathBuf {
    let mut s = store.as_os_str().to_owned();
    s.push(".key");
    PathBuf::from(s)
}

/// Tighten a keyfile's permissions to owner-only where the OS models it (unix `0600`).
/// On Windows we rely on the profile/directory ACLs — `std` exposes no portable mode.
#[cfg(unix)]
fn restrict_perms(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}
#[cfg(not(unix))]
fn restrict_perms(_path: &Path) {}

/// Generate a fresh random 32-byte key, persist it to `path`, and return it.
fn generate_keyfile(path: &Path) -> Vec<u8> {
    let mut k = [0u8; 32];
    getrandom::getrandom(&mut k).unwrap_or_else(|_| die("keygen: system entropy unavailable"));
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(path, k)
        .unwrap_or_else(|e| die(&format!("write keyfile {}: {e}", path.display())));
    restrict_perms(path);
    k.to_vec()
}

/// The per-store key: `$MNEMA_KEY` if set, else the sidecar keyfile. A keyfile is generated
/// only for a store that does not yet exist — never for an existing store missing its key
/// (that is a migration, handled by `rekey`), so we can't silently lock the data away.
fn resolve_key(store: &Path) -> Vec<u8> {
    if let Ok(k) = std::env::var("MNEMA_KEY")
        && !k.is_empty()
    {
        return k.into_bytes();
    }
    let keyfile = keyfile_path(store);
    match std::fs::read(&keyfile) {
        Ok(b) if b.len() == 32 => b,
        Ok(_) => die(&format!(
            "keyfile {} is malformed (expected 32 bytes)",
            keyfile.display()
        )),
        Err(_) if store.exists() => {
            die("store exists but has no keyfile and $MNEMA_KEY is unset — \
             set $MNEMA_KEY to the old passphrase and run `mnema rekey <store>` to migrate")
        }
        Err(_) => generate_keyfile(&keyfile),
    }
}

fn load(store: &str) -> Mnema<HashEmbedder> {
    let embedder = HashEmbedder::new(DIMS);
    let path = Path::new(store);
    if path.exists() {
        let bytes = std::fs::read(store).unwrap_or_else(|e| die(&format!("read {store}: {e}")));
        Mnema::open(&bytes, &resolve_key(path), embedder)
            .unwrap_or_else(|_| die("cannot open store (wrong key or corrupt)"))
    } else {
        Mnema::new(embedder)
    }
}

fn save(store: &str, mem: &mut Mnema<HashEmbedder>) {
    let blob = mem
        .seal(&resolve_key(Path::new(store)))
        .unwrap_or_else(|_| die("seal failed"));
    std::fs::write(store, blob).unwrap_or_else(|e| die(&format!("write {store}: {e}")));
}

/// Migrate a store to a per-store keyfile: open it with the current `$MNEMA_KEY`, then
/// re-seal it under a freshly generated `<store>.key`. Refuses to clobber an existing
/// keyfile, so it is safe to run at most once per store.
fn rekey(store: &str) {
    let path = Path::new(store);
    if !path.exists() {
        die(&format!("rekey: store {store} does not exist"));
    }
    let old = match std::env::var("MNEMA_KEY") {
        Ok(k) if !k.is_empty() => k.into_bytes(),
        _ => die("rekey: set $MNEMA_KEY to the store's CURRENT passphrase"),
    };
    let keyfile = keyfile_path(path);
    if keyfile.exists() {
        die(&format!(
            "rekey: {} already exists; refusing to overwrite",
            keyfile.display()
        ));
    }
    let bytes = std::fs::read(store).unwrap_or_else(|e| die(&format!("read {store}: {e}")));
    let mut mem = Mnema::open(&bytes, &old, HashEmbedder::new(DIMS))
        .unwrap_or_else(|_| die("rekey: cannot open store with $MNEMA_KEY (wrong passphrase?)"));
    let new_key = generate_keyfile(&keyfile);
    let blob = mem
        .seal(&new_key)
        .unwrap_or_else(|_| die("rekey: seal failed"));
    std::fs::write(store, blob).unwrap_or_else(|e| die(&format!("write {store}: {e}")));
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
            let mut mem = load(store);
            let id = mem.remember(tier(&args[2]), &args[3]);
            save(store, &mut mem);
            println!("{id}");
        }
        ("fact", 5) => {
            let store = &args[1];
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
            let mem = load(&args[1]);
            println!("memories: {}  indexed: {}", mem.len(), mem.indexed());
        }
        ("rekey", 2) => rekey(&args[1]),
        _ => die(
            "usage: mnema remember|recall|fact|stats|rekey <store> ...  (see the source header)",
        ),
    }
}
