//! Spawned-process proof of the `mnema` CLI's end-to-end behavior: the store write-lock
//! (directory creation, lock-file handling, contention refusal), `rekey` (existence check,
//! passphrase requirement, keyfile resume, malformed-keyfile refusal), `reinforce` (both sides
//! of the found/not-found branch), and `forget-fact` (subject + optional attribute filtering).
//!
//! Everything here sits behind `main()` and `die()` — which call `std::process::exit` — so it
//! can only be observed by driving the real binary with a controlled environment and asserting
//! exit codes, stdout, stderr, and on-disk state.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// The passphrase used wherever a test wants an env-provided key.
const KEY: &str = "cli-integration-test-passphrase";

/// A unique temp directory, removed on drop, so parallel tests and reruns never collide.
struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(label: &str) -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!("mnema_cli_{label}_{}", std::process::id()));
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

/// Run the real `mnema` binary with `args`, `MNEMA_KEY` scrubbed from the ambient environment
/// and, if given, set to `key`. The CLI reads no stdin and exits on its own.
fn mnema(key: Option<&str>, args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mnema"));
    cmd.args(args).env_remove("MNEMA_KEY").stdin(Stdio::null());
    if let Some(k) = key {
        cmd.env("MNEMA_KEY", k);
    }
    cmd.output().expect("spawn the mnema binary")
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The sidecar keyfile path for a store: `<store>.key` (mirrors `keyfile::keyfile_path`).
fn sidecar(store: &Path) -> PathBuf {
    let mut s = store.as_os_str().to_owned();
    s.push(".key");
    PathBuf::from(s)
}

#[test]
fn remember_creates_the_store_directory_and_prints_the_new_id() {
    let td = TempDirGuard::new("mkdir");
    // The store lives in a directory that does NOT exist yet: lock_store must create it, then
    // take the lock, and the command must succeed printing the new memory's numeric id.
    let store = td.0.join("nested").join("store.mn");
    let s = store.to_str().unwrap();
    let out = mnema(
        Some(KEY),
        &["remember", s, "open", "the cat sat on the mat"],
    );
    assert!(
        out.status.success(),
        "remember into a not-yet-created directory must succeed: {}",
        stderr_of(&out)
    );
    let id_line = stdout_of(&out);
    id_line
        .trim()
        .parse::<u64>()
        .expect("remember prints the new memory id");
    assert!(store.exists(), "the store must have been written");
}

#[test]
fn a_write_command_preserves_existing_lock_file_content() {
    let td = TempDirGuard::new("locktrunc");
    let store = td.0.join("store.mn");
    let s = store.to_str().unwrap();
    // The lock file is opened with truncate(false): pre-existing content must survive a write
    // command taking and releasing the lock.
    let lockpath = format!("{s}.lock");
    std::fs::write(&lockpath, b"sentinel").unwrap();
    let out = mnema(Some(KEY), &["remember", s, "open", "hello"]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        std::fs::read(&lockpath).unwrap(),
        b"sentinel",
        "taking the store lock must not truncate the lock file"
    );
}

#[test]
fn a_second_writer_is_refused_while_the_store_lock_is_held() {
    let td = TempDirGuard::new("contend");
    let store = td.0.join("store.mn");
    let s = store.to_str().unwrap();
    // Hold the advisory lock the way the CLI does (same `<store>.lock` sibling), then spawn a
    // write command: it must refuse with the in-use message and a nonzero exit, never proceed.
    let lockpath = format!("{s}.lock");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lockpath)
        .unwrap();
    lock.lock().expect("take the exclusive advisory lock");
    let out = mnema(Some(KEY), &["remember", s, "open", "must not land"]);
    assert!(
        !out.status.success(),
        "a write command must be refused while the lock is held (stdout: {})",
        stdout_of(&out)
    );
    assert!(
        stderr_of(&out).contains("is in use by another mnema process"),
        "the refusal must say the store is in use: {}",
        stderr_of(&out)
    );
    assert!(
        !store.exists(),
        "the refused writer must not touch the store"
    );
    drop(lock);
}

#[test]
fn rekey_reseals_an_existing_store_under_a_fresh_keyfile() {
    let td = TempDirGuard::new("rekey");
    let store = td.0.join("store.mn");
    let s = store.to_str().unwrap();
    // A store sealed under an env passphrase (no sidecar keyfile yet)...
    let out = mnema(
        Some(KEY),
        &["remember", s, "open", "carried across the rekey"],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert!(!sidecar(&store).exists());
    // ...rekeys under a freshly generated `<store>.key`...
    let out = mnema(Some(KEY), &["rekey", s]);
    assert!(
        out.status.success(),
        "rekey of an existing store under the correct $MNEMA_KEY must succeed: {}",
        stderr_of(&out)
    );
    assert!(stdout_of(&out).contains("rekeyed"), "{}", stdout_of(&out));
    let kf = sidecar(&store);
    assert!(kf.exists(), "rekey must generate the sidecar keyfile");
    assert_eq!(std::fs::read(&kf).unwrap().len(), 32);
    // ...after which the store opens via the keyfile alone (no $MNEMA_KEY).
    let out = mnema(None, &["stats", s]);
    assert!(
        out.status.success(),
        "the rekeyed store must open under the new keyfile: {}",
        stderr_of(&out)
    );
    assert!(
        stdout_of(&out).contains("memories: 1"),
        "{}",
        stdout_of(&out)
    );
}

#[test]
fn rekey_refuses_a_missing_store() {
    let td = TempDirGuard::new("rekey_missing");
    let store = td.0.join("no-such.mn");
    let s = store.to_str().unwrap();
    let out = mnema(Some(KEY), &["rekey", s]);
    assert!(!out.status.success(), "rekey of a missing store must fail");
    assert!(
        stderr_of(&out).contains("does not exist"),
        "the failure must name the actual problem (a missing store), not a later I/O error: {}",
        stderr_of(&out)
    );
}

#[test]
fn rekey_resumes_under_an_existing_keyfile_and_rejects_a_malformed_one() {
    // Resume: an existing, well-formed 32-byte keyfile (the interrupted-rekey artifact) is
    // REUSED, and the store ends up sealed under exactly that key.
    let td = TempDirGuard::new("rekey_resume");
    let store = td.0.join("store.mn");
    let s = store.to_str().unwrap();
    let out = mnema(Some(KEY), &["remember", s, "open", "survives the resume"]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let kf = sidecar(&store);
    std::fs::write(&kf, [7u8; 32]).unwrap();
    let out = mnema(Some(KEY), &["rekey", s]);
    assert!(
        out.status.success(),
        "rekey must resume with an existing 32-byte keyfile: {}",
        stderr_of(&out)
    );
    assert_eq!(
        std::fs::read(&kf).unwrap(),
        [7u8; 32],
        "resume must reuse the existing keyfile, not overwrite it"
    );
    let out = mnema(None, &["stats", s]);
    assert!(
        out.status.success(),
        "the store must open under the resumed keyfile: {}",
        stderr_of(&out)
    );
    assert!(
        stdout_of(&out).contains("memories: 1"),
        "{}",
        stdout_of(&out)
    );

    // Refusal: a keyfile of the wrong length is malformed — rekey must refuse rather than
    // seal the store under garbage.
    let store2 = td.0.join("store2.mn");
    let s2 = store2.to_str().unwrap();
    let out = mnema(Some(KEY), &["remember", s2, "open", "second store"]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    std::fs::write(sidecar(&store2), [7u8; 5]).unwrap();
    let out = mnema(Some(KEY), &["rekey", s2]);
    assert!(
        !out.status.success(),
        "rekey must refuse a keyfile that is not exactly 32 bytes"
    );
    assert!(stderr_of(&out).contains("malformed"), "{}", stderr_of(&out));
}

#[test]
fn reinforce_strengthens_an_existing_id_and_refuses_an_unknown_one() {
    let td = TempDirGuard::new("reinforce");
    let store = td.0.join("store.mn");
    let s = store.to_str().unwrap();
    let out = mnema(Some(KEY), &["remember", s, "open", "worth reinforcing"]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let id = stdout_of(&out).trim().to_string();

    // TRUE side: a real id is reinforced and reported.
    let out = mnema(Some(KEY), &["reinforce", s, &id]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert!(
        stdout_of(&out).contains(&format!("reinforced {id}")),
        "{}",
        stdout_of(&out)
    );

    // FALSE side: an unknown id is refused with a nonzero exit and the not-found message.
    let out = mnema(Some(KEY), &["reinforce", s, "999999999"]);
    assert!(
        !out.status.success(),
        "reinforcing a nonexistent id must fail (stdout: {})",
        stdout_of(&out)
    );
    assert!(
        stderr_of(&out).contains("no memory with id"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn forget_fact_matches_subject_and_optionally_attribute() {
    let td = TempDirGuard::new("forget_fact");
    let store = td.0.join("store.mn");
    let s = store.to_str().unwrap();
    for [subject, attribute, value] in [
        ["alice", "color", "red"],
        ["alice", "size", "big"],
        ["bob", "color", "blue"],
    ] {
        let out = mnema(Some(KEY), &["fact", s, subject, attribute, value]);
        assert!(out.status.success(), "{}", stderr_of(&out));
    }

    // With an attribute: exactly alice.color goes — alice.size and bob.color both stay.
    let out = mnema(Some(KEY), &["forget-fact", s, "alice", "color"]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert!(
        stdout_of(&out).contains("forgot 1 belief record(s)"),
        "forgetting alice.color must remove exactly one record: {}",
        stdout_of(&out)
    );
    let alice = stdout_of(&mnema(Some(KEY), &["beliefs", s, "alice"]));
    assert!(alice.contains("alice.size = big"), "{alice}");
    assert!(
        !alice.contains("alice.color"),
        "alice.color must be gone: {alice}"
    );
    let bob = stdout_of(&mnema(Some(KEY), &["beliefs", s, "bob"]));
    assert!(bob.contains("bob.color = blue"), "{bob}");

    // Without an attribute: everything about bob goes — alice's remaining belief stays.
    let out = mnema(Some(KEY), &["forget-fact", s, "bob"]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert!(
        stdout_of(&out).contains("forgot 1 belief record(s)"),
        "forgetting all of bob must remove exactly his one record: {}",
        stdout_of(&out)
    );
    let bob = stdout_of(&mnema(Some(KEY), &["beliefs", s, "bob"]));
    assert!(
        !bob.contains("bob."),
        "bob must have no beliefs left: {bob}"
    );
    let alice = stdout_of(&mnema(Some(KEY), &["beliefs", s, "alice"]));
    assert!(alice.contains("alice.size = big"), "{alice}");
}
