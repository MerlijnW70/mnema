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

/// Tighten a keyfile's permissions to owner-only where the OS models it (unix `0600`). On Windows
/// we rely on the profile/directory ACLs — `std` exposes no portable mode.
#[cfg(unix)]
fn restrict_perms(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}
#[cfg(not(unix))]
fn restrict_perms(_path: &Path) {}

/// A fresh, strong random passphrase — hex of 32 random bytes (256 bits) — suitable for use as
/// `$MNEMA_KEY`. For callers who want a portable, human-copyable secret (a shared store, CI, an MCP
/// client config) rather than the on-disk sidecar keyfile.
pub fn generate_passphrase() -> Result<String, KeyError> {
    let mut k = [0u8; 32];
    getrandom::getrandom(&mut k).map_err(|_| KeyError::Entropy)?;
    Ok(k.iter().map(|b| format!("{b:02x}")).collect())
}

/// Generate a fresh random 32-byte key, persist it to `path` (owner-only where modelled), return it.
pub fn generate_keyfile(path: &Path) -> Result<Vec<u8>, KeyError> {
    let mut k = [0u8; 32];
    getrandom::getrandom(&mut k).map_err(|_| KeyError::Entropy)?;
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(path, k).map_err(|e| KeyError::WriteFailed(path.to_path_buf(), e))?;
    restrict_perms(path);
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
