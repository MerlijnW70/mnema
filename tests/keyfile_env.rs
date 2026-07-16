//! Spawned-process proof of `$MNEMA_KEY` handling in `keyfile::resolve_key`: a set, NON-EMPTY
//! `$MNEMA_KEY` is the key (no sidecar keyfile is ever generated), while an EMPTY one is no
//! passphrase at all and falls back to the generated sidecar keyfile. Env-dependent behavior
//! can't be tested in-process (`set_var` is unsafe in edition 2024 and the crate forbids unsafe),
//! so each case drives the real `mnema` binary with a controlled environment and asserts the
//! observable file-state outcome.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// A unique temp directory, removed on drop, so parallel tests and reruns never collide.
struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(label: &str) -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!("mnema_keyfile_env_{label}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The sidecar keyfile path for a store: `<store>.key` (mirrors `keyfile::keyfile_path`).
fn sidecar(store: &Path) -> PathBuf {
    let mut s = store.as_os_str().to_owned();
    s.push(".key");
    PathBuf::from(s)
}

/// Run `mnema remember <store> private <content>` with `MNEMA_KEY` scrubbed from the ambient
/// environment and, if given, set to `key`. The CLI reads no stdin and exits on its own.
fn run_remember(store: &Path, key: Option<&str>) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mnema"));
    cmd.arg("remember")
        .arg(store)
        .arg("private")
        .arg("the cat sat on the mat")
        .env_remove("MNEMA_KEY")
        .stdin(Stdio::null());
    if let Some(k) = key {
        cmd.env("MNEMA_KEY", k);
    }
    cmd.output().expect("spawn the mnema binary")
}

#[test]
fn a_non_empty_mnema_key_is_the_key_and_no_sidecar_keyfile_is_generated() {
    let td = TempDirGuard::new("set");
    let store = td.0.join("store.mn");
    let out = run_remember(&store, Some("integration-test-passphrase"));
    assert!(
        out.status.success(),
        "remember must succeed under $MNEMA_KEY: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(store.exists(), "the store must have been written");
    assert!(
        !sidecar(&store).exists(),
        "a set, non-empty $MNEMA_KEY IS the key — generating a sidecar keyfile anyway would \
         seal the store under a key the user never chose"
    );
}

#[test]
fn an_empty_mnema_key_is_no_passphrase_and_falls_back_to_a_generated_sidecar() {
    let td = TempDirGuard::new("empty");
    let store = td.0.join("store.mn");
    let out = run_remember(&store, Some(""));
    assert!(
        out.status.success(),
        "remember must succeed with an empty $MNEMA_KEY: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let kf = sidecar(&store);
    assert!(
        kf.exists(),
        "an empty $MNEMA_KEY is not a passphrase — the sidecar keyfile must be generated, \
         never an empty-key seal"
    );
    assert_eq!(
        std::fs::read(&kf).unwrap().len(),
        32,
        "the generated sidecar key is exactly 32 bytes"
    );
}
