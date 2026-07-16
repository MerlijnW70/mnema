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
    /// The sidecar keyfile (or the store's own metadata) could not be READ — a permission or
    /// sharing violation, not absence. Generating a fresh key here would atomically rename it
    /// over a keyfile that still exists, destroying the only key to the store; refuse instead.
    ReadFailed(PathBuf, std::io::Error),
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
            KeyError::ReadFailed(p, e) => {
                write!(
                    f,
                    "could not read {}: {e} — refusing to generate a fresh key while the \
                     existing one may still be there. Resolve the I/O error and retry.",
                    p.display()
                )
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

/// On Windows there is no mode bit, but there ARE ACLs — and a store outside the user profile
/// (a shared folder, C:\Temp) hands the keyfile down whatever read access that directory grants,
/// including BUILTIN\Users on many non-profile paths: any local user could read the raw store
/// key and decrypt the store offline. So: create the file, then cut its ACL down to owner-only
/// via `icacls` (strip inheritance, grant only the file's owner — `*S-1-3-4` is the OWNER RIGHTS
/// SID, locale-independent) **before any key byte is written**. A failed restriction fails the
/// open — an unprotected keyfile must never be created silently.
#[cfg(windows)]
fn open_private_new(path: &Path) -> std::io::Result<std::fs::File> {
    open_private_new_with(path, restrict_to_owner)
}

/// The `icacls` invocation itself, split out so tests can drive the failure branch of
/// [`open_private_new_with`] with a deterministic failing restrictor (a real `icacls` failure
/// cannot be provoked hermetically on a file we just created).
#[cfg(windows)]
fn restrict_to_owner(path: &Path) -> std::io::Result<std::process::ExitStatus> {
    std::process::Command::new("icacls")
        .arg(path)
        .args(["/inheritance:r", "/grant:r", "*S-1-3-4:F"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
}

#[cfg(windows)]
fn open_private_new_with(
    path: &Path,
    restrict: fn(&Path) -> std::io::Result<std::process::ExitStatus>,
) -> std::io::Result<std::fs::File> {
    let _ = std::fs::remove_file(path);
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let status = restrict(path)?;
    if !status.success() {
        drop(f);
        let _ = std::fs::remove_file(path);
        return Err(std::io::Error::other(
            "icacls could not restrict the keyfile to owner-only",
        ));
    }
    Ok(f)
}

// No third platform: `unix` (Linux/macOS/BSD, `0600`) and `windows` (icacls owner-only ACL) are
// the entire support matrix for the `secure` feature. Rather than carry an unprivileged fallback
// that would silently create a world-readable keyfile — and that no test on any supported target
// could ever exercise (so a mutation of it can never be killed) — refuse to build. A new target
// must add a real owner-only `open_private_new` here.
#[cfg(not(any(unix, windows)))]
compile_error!(
    "mnema's `secure` feature supports only unix and windows targets: the keyfile holding the \
     raw store key must be created owner-only, and no such mechanism is defined for this target."
);

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
/// (see `write_keyfile_atomic`), and return it. A write/permission failure is surfaced as
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
        // ONLY a genuinely absent keyfile may fall through toward generation. Any other read
        // error (permissions, a sharing violation, transient I/O) means the keyfile may still
        // exist — and generate_keyfile would atomically rename a FRESH key over it, destroying
        // the only key to the store. Fail closed instead.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // The store-existence probe must fail closed the same way: `Path::exists()` reads a
            // stat *error* as "no store", which would invent a fresh key for a store that does
            // exist — sealing future writes under a key its data was never written with.
            match store.try_exists() {
                Ok(true) => Err(KeyError::ExistingStoreNoKeyfile),
                Ok(false) => generate_keyfile(&keyfile),
                Err(e) => Err(KeyError::ReadFailed(store.to_path_buf(), e)),
            }
        }
        Err(e) => Err(KeyError::ReadFailed(keyfile, e)),
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

    #[test]
    fn resolve_key_fails_closed_when_the_keyfile_is_unreadable_rather_than_regenerating() {
        if !env_key_unset() {
            return;
        }
        // The key-destruction trap: a keyfile that exists but cannot be READ (permissions, a
        // sharing violation) must never be treated like an absent one — generation would rename
        // a FRESH key over it and the store would be locked away forever. A directory at the
        // keyfile path is the portable "read fails, but NOT with NotFound" stand-in.
        let ts = TempStore::new("unreadable");
        let kf = keyfile_path(&ts.0);
        std::fs::create_dir_all(&kf).unwrap();
        let r = resolve_key(&ts.0);
        assert!(
            matches!(r, Err(KeyError::ReadFailed(_, _))),
            "an unreadable keyfile must refuse, not regenerate: {r:?}"
        );
        assert!(kf.is_dir(), "nothing may be written over the keyfile path");
        let _ = std::fs::remove_dir_all(&kf);
    }

    #[test]
    fn generate_keyfile_creates_missing_parent_directories() {
        // A keyfile path inside a directory that does not exist yet must be created, not fail:
        // the non-empty parent has to survive the `.filter(|d| !d.as_os_str().is_empty())`
        // guard and reach `create_dir_all`.
        let mut root = std::env::temp_dir();
        root.push(format!("mnema_keyfile_test_nested_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let kf = root.join("a").join("b").join("store.key");
        let k = generate_keyfile(&kf).unwrap();
        assert_eq!(
            std::fs::read(&kf).unwrap(),
            k,
            "the key must land inside the freshly created parent directories"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(windows)]
    #[test]
    fn open_private_new_fails_and_removes_the_file_when_restriction_fails() {
        // The unprotected-keyfile trap: if the ACL restriction fails, the open must FAIL and the
        // file must be GONE — never a silently world-readable keyfile. Driven through a
        // deterministic failing restrictor; the success side is asserted right after.
        fn failing(_: &Path) -> std::io::Result<std::process::ExitStatus> {
            std::process::Command::new("cmd")
                .args(["/C", "exit", "1"])
                .status()
        }
        fn passing(_: &Path) -> std::io::Result<std::process::ExitStatus> {
            std::process::Command::new("cmd")
                .args(["/C", "exit", "0"])
                .status()
        }
        let ts = TempStore::new("winrestrict");
        let kf = keyfile_path(&ts.0);
        let r = open_private_new_with(&kf, failing);
        assert!(r.is_err(), "a failed ACL restriction must fail the open");
        assert!(
            !kf.exists(),
            "an unprotected keyfile must never be left behind"
        );
        let f = open_private_new_with(&kf, passing).expect("a successful restriction opens");
        drop(f);
        assert!(kf.exists(), "the successfully restricted file must remain");
    }

    #[cfg(windows)]
    #[test]
    fn generate_keyfile_is_acl_restricted_to_the_owner_on_windows() {
        // The Windows mirror of the unix 0600 test: the keyfile's ACL must carry NO inherited
        // ACEs (no "(I)" markers — those are what hand BUILTIN\Users read access on non-profile
        // paths) and exactly ONE explicit ACE, the owner-only grant.
        let ts = TempStore::new("winacl");
        let kf = keyfile_path(&ts.0);
        generate_keyfile(&kf).unwrap();
        let out = std::process::Command::new("icacls")
            .arg(&kf)
            .output()
            .expect("icacls is a Windows system binary");
        let listing = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(
            !listing.contains("(I)"),
            "the keyfile must not inherit directory ACEs: {listing}"
        );
        assert_eq!(
            listing.matches(":(").count(),
            1,
            "exactly one explicit owner-only ACE: {listing}"
        );
    }
}
