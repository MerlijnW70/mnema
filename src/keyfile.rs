//! Per-store key resolution, shared by the `mnema` CLI and the `mnema-server` server.
//!
//! The key is per-store, resolved in this order (never on the command line):
//!   1. `$MNEMA_KEY` if set and non-empty — an explicit passphrase (shared stores, CI, env-only
//!      secrets);
//!   2. else a random 32-byte key in the sidecar `<store>.key`, generated on first use for a store
//!      that does not yet exist.
//!
//! There is no shared default and no empty-passphrase fallback: a store is always sealed under a
//! real key. This is I/O glue **below the behavioral waterline** (not in the probed `sources`), and
//! it lives behind the `secure` feature so the zero-dependency wasm core never pulls in `std::fs`.

use std::path::{Path, PathBuf};

use zeroize::Zeroizing;

/// Why [`resolve_key`] could not produce a key. Callers render and exit as they see fit.
#[derive(Debug)]
pub enum KeyError {
    /// The sidecar keyfile exists but is not exactly 32 bytes.
    MalformedKeyfile(PathBuf),
    /// The store exists but has no keyfile and `$MNEMA_KEY` is unset — a migration, not a fresh
    /// store, so inventing a key here would seal a *new* store over data that could never be
    /// opened again. The caller should point the user at `rekey`.
    ExistingStoreNoKeyfile,
    /// The OS entropy source failed while generating a fresh key.
    Entropy,
    /// Persisting the sidecar keyfile failed.
    WriteFailed(PathBuf, std::io::Error),
}

impl std::fmt::Display for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyError::MalformedKeyfile(p) => {
                write!(
                    f,
                    "keyfile {} is malformed (expected 32 bytes)",
                    p.display()
                )
            }
            KeyError::ExistingStoreNoKeyfile => write!(
                f,
                "store exists but has no keyfile and $MNEMA_KEY is unset — set $MNEMA_KEY to the \
                 store's passphrase (or, if a rekey was interrupted, the OLD one and re-run `rekey`)"
            ),
            KeyError::Entropy => write!(f, "system entropy unavailable while generating a key"),
            KeyError::WriteFailed(p, e) => {
                write!(f, "could not write keyfile {}: {e}", p.display())
            }
        }
    }
}

impl std::error::Error for KeyError {}

/// The sidecar keyfile path for a store: `<store>.key`.
pub fn keyfile_path(store: &Path) -> PathBuf {
    let mut s = store.as_os_str().to_owned();
    s.push(".key");
    PathBuf::from(s)
}

/// Persist `key` to `path` **atomically, durably, and owner-only**: write a sibling `.tmp`
/// created with restrictive permissions and `O_EXCL` (so the key is never briefly world-readable
/// and we never follow a pre-planted symlink), fsync it, atomically rename it over `path`, then
/// fsync the parent directory. Durability matters beyond tidiness: `rekey` seals the store under
/// this key and fsyncs *that*, so if the keyfile were only in the page cache a power loss could
/// leave a store sealed under a key that never reached disk — permanently unrecoverable.
fn write_keyfile_atomic(path: &Path, key: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);

    {
        let mut f = open_private_new(&tmp)?;
        f.write_all(key)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    sync_parent_dir(path);
    Ok(())
}

/// Open `path` for writing, creating it new with owner-only permissions where the OS models them
/// (unix `0600`). `create_new` + a prior unlink of any stale temp means we never follow a
/// symlink planted at the temp path. On Windows there is no portable mode bit, so we rely on the
/// profile/directory ACLs — but the create-new-then-rename shape still holds.
#[cfg(unix)]
fn open_private_new(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let _ = std::fs::remove_file(path); // clear a stale temp from an interrupted write
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_private_new(path: &Path) -> std::io::Result<std::fs::File> {
    let _ = std::fs::remove_file(path);
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

/// fsync the directory holding `path` so a rename's directory entry is durable (POSIX). Windows
/// has no directory handle to fsync here; the temp fsync + atomic replace already avoid a torn
/// file, so this is a best-effort no-op there.
#[cfg(unix)]
fn sync_parent_dir(path: &Path) {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty())
        && let Ok(d) = std::fs::File::open(dir)
    {
        let _ = d.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) {}

/// A fresh, strong random passphrase — hex of 32 random bytes (256 bits) — suitable for use as
/// `$MNEMA_KEY`. For callers who want a portable, human-copyable secret (a shared store, CI, an MCP
/// client config) rather than the on-disk sidecar keyfile.
pub fn generate_passphrase() -> Result<String, KeyError> {
    let mut k = Zeroizing::new([0u8; 32]);
    getrandom::getrandom(k.as_mut_slice()).map_err(|_| KeyError::Entropy)?;
    Ok(k.iter().map(|b| format!("{b:02x}")).collect())
}

/// Generate a fresh random 32-byte key, persist it to `path` atomically, durably, and owner-only
/// (see [`write_keyfile_atomic`]), and return it. A write/permission failure is surfaced as
/// [`KeyError::WriteFailed`], never silently ignored, so a store is never sealed under a key that
/// did not actually reach disk.
pub fn generate_keyfile(path: &Path) -> Result<Vec<u8>, KeyError> {
    let mut k = Zeroizing::new([0u8; 32]);
    getrandom::getrandom(k.as_mut_slice()).map_err(|_| KeyError::Entropy)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| KeyError::WriteFailed(path.to_path_buf(), e))?;
    }
    write_keyfile_atomic(path, k.as_slice())
        .map_err(|e| KeyError::WriteFailed(path.to_path_buf(), e))?;
    Ok(k.to_vec())
}

/// The per-store key: `$MNEMA_KEY` if set and non-empty, else the sidecar keyfile. A keyfile is
/// generated only for a store that does not yet exist — never for an existing store missing its
/// key (that is a migration, handled by `rekey`), so we can't silently lock the data away.
pub fn resolve_key(store: &Path) -> Result<Vec<u8>, KeyError> {
    if let Ok(k) = std::env::var("MNEMA_KEY")
        && !k.is_empty()
    {
        return Ok(k.into_bytes());
    }
    let keyfile = keyfile_path(store);
    match std::fs::read(&keyfile) {
        Ok(b) if b.len() == 32 => Ok(b),
        Ok(_) => Err(KeyError::MalformedKeyfile(keyfile)),
        Err(_) if store.exists() => Err(KeyError::ExistingStoreNoKeyfile),
        Err(_) => generate_keyfile(&keyfile),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp store path (with sidecar/tmp) cleaned up on drop. File-based only — these
    /// tests never touch `$MNEMA_KEY`, so they don't race the process-global environment.
    struct TempStore(PathBuf);
    impl TempStore {
        fn new(label: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("mnema_keyfile_test_{label}"));
            let ts = Self(p);
            ts.cleanup();
            ts
        }
        fn cleanup(&self) {
            let _ = std::fs::remove_file(&self.0);
            let kf = keyfile_path(&self.0);
            let mut tmp = kf.clone().into_os_string();
            tmp.push(".tmp");
            let _ = std::fs::remove_file(&kf);
            let _ = std::fs::remove_file(PathBuf::from(tmp));
        }
    }
    impl Drop for TempStore {
        fn drop(&mut self) {
            self.cleanup();
        }
    }

    /// These tests assert the sidecar-key path, which only runs when `$MNEMA_KEY` is unset (the
    /// server's normal mode). If the ambient environment has it set, skip rather than false-fail.
    fn env_key_unset() -> bool {
        std::env::var_os("MNEMA_KEY").is_none()
    }

    #[test]
    fn generate_keyfile_writes_a_32_byte_key_and_leaves_no_temp() {
        let ts = TempStore::new("generate");
        let kf = keyfile_path(&ts.0);
        let k = generate_keyfile(&kf).unwrap();
        assert_eq!(k.len(), 32);
        assert_eq!(
            std::fs::read(&kf).unwrap(),
            k,
            "persisted bytes match the returned key"
        );
        let mut tmp = kf.into_os_string();
        tmp.push(".tmp");
        assert!(
            !Path::new(&tmp).exists(),
            "the atomic write must rename the temp away, never leave it"
        );
    }

    #[cfg(unix)]
    #[test]
    fn generate_keyfile_is_created_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let ts = TempStore::new("perms");
        let kf = keyfile_path(&ts.0);
        generate_keyfile(&kf).unwrap();
        let mode = std::fs::metadata(&kf).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "the key must be owner-only from creation, never briefly world-readable"
        );
    }

    #[test]
    fn resolve_key_generates_then_reads_back_the_same_sidecar_key() {
        if !env_key_unset() {
            return;
        }
        let ts = TempStore::new("resolve_reuse");
        let first = resolve_key(&ts.0).unwrap();
        assert_eq!(first.len(), 32);
        // A second resolve reads the SAME sidecar back, not a fresh key.
        assert_eq!(resolve_key(&ts.0).unwrap(), first);
    }

    #[test]
    fn resolve_key_refuses_an_existing_store_missing_its_keyfile() {
        if !env_key_unset() {
            return;
        }
        let ts = TempStore::new("existing_no_key");
        // A store file exists but its keyfile is gone (an interrupted migration): inventing a key
        // here would seal a NEW store over data that could never be opened again.
        std::fs::write(&ts.0, b"pretend-sealed-store").unwrap();
        assert!(matches!(
            resolve_key(&ts.0),
            Err(KeyError::ExistingStoreNoKeyfile)
        ));
    }

    #[test]
    fn resolve_key_rejects_a_wrong_length_keyfile() {
        if !env_key_unset() {
            return;
        }
        let ts = TempStore::new("malformed");
        std::fs::write(keyfile_path(&ts.0), b"not-thirty-two-bytes").unwrap();
        assert!(matches!(
            resolve_key(&ts.0),
            Err(KeyError::MalformedKeyfile(_))
        ));
    }
}
