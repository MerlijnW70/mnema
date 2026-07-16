//! Spawned-process proof of the `mnema-server` binary's launch behavior: config resolution
//! (`--path` beats `$MNEMA_PATH`; `--local` / `$MNEMA_LOCAL` are the ONLY ways to open the
//! egress wall, and `$MNEMA_LOCAL=0` keeps it closed), `--migrate` refusing to serve, the
//! store lock refusing a second server, and refusal to overwrite a corrupt store.
//!
//! Everything here sits behind `main()` / `std::process::exit` and reads argv + environment,
//! so it is only observable by driving the real binary over piped stdio with line-delimited
//! JSON-RPC and asserting replies, exit codes, stderr, and on-disk state.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// The passphrase every spawned server seals with (`$MNEMA_KEY`), so reruns are deterministic
/// and no sidecar keyfile logic is in play.
const KEY: &str = "server-integration-test-passphrase";

/// A hung (or mutated-into-hanging) child is killed after this long, so it can't hang the suite.
const TIMEOUT: Duration = Duration::from_secs(60);

/// A unique temp directory, removed on drop, so parallel tests and reruns never collide.
struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(label: &str) -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!("mnema_server_{label}_{}", std::process::id()));
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

/// The one process-wide stub embeddings endpoint (see [`stub_embed_endpoint`]) — an
/// `http-embed` build of the server probes `$MNEMA_EMBED_URL` at startup, so every spawn
/// needs something local to talk to. A non-`http-embed` build simply ignores it.
fn embed_url() -> &'static str {
    static STUB: OnceLock<String> = OnceLock::new();
    STUB.get_or_init(stub_embed_endpoint)
}

/// A minimal loopback embeddings server (Ollama shape) answering every request with a fixed
/// width-8 vector — one request per connection, `Connection: close`. The thread serves for the
/// life of the test process; reads are timeout-guarded so a wedged client can't hang it.
fn stub_embed_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind the stub embeddings listener");
    let port = listener.local_addr().expect("stub local addr").port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
            // Read the full request: headers, then exactly Content-Length body bytes.
            let mut buf = Vec::new();
            let mut tmp = [0_u8; 1024];
            let mut header_end: Option<usize> = None;
            let mut content_len = 0_usize;
            loop {
                match stream.read(&mut tmp) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                }
                if header_end.is_none()
                    && let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n")
                {
                    header_end = Some(pos + 4);
                    for line in String::from_utf8_lossy(&buf[..pos]).lines() {
                        let lower = line.to_ascii_lowercase();
                        if let Some(v) = lower.strip_prefix("content-length:") {
                            content_len = v.trim().parse().unwrap_or(0);
                        }
                    }
                }
                if let Some(end) = header_end
                    && buf.len() >= end + content_len
                {
                    break;
                }
            }
            let body = r#"{"embedding":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}/api/embeddings")
}

/// A `mnema-server` command with piped stdio and a scrubbed-then-controlled environment: the
/// ambient `MNEMA_*` variables are removed, `MNEMA_KEY` is pinned to [`KEY`], the embedder is
/// pointed at the local stub endpoint, and only the given `envs` are added back.
fn server_cmd(args: &[&str], envs: &[(&str, &str)]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mnema-server"));
    cmd.args(args)
        .env_remove("MNEMA_PATH")
        .env_remove("MNEMA_LOCAL")
        .env_remove("MNEMA_KEY")
        .env_remove("MNEMA_EMBED_URL")
        .env_remove("MNEMA_EMBED_MODEL")
        .env_remove("MNEMA_EMBED_API")
        .env_remove("MNEMA_EMBED_ALLOW_REMOTE")
        .env("MNEMA_KEY", KEY)
        .env("MNEMA_EMBED_URL", embed_url())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd
}

/// Wait for `child` with a deadline, killing it if [`TIMEOUT`] passes — the guard that keeps a
/// hung server from hanging the whole suite.
fn wait_with_deadline(child: &mut Child) -> ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("wait on mnema-server") {
            return status;
        }
        if start.elapsed() > TIMEOUT {
            let _ = child.kill();
            return child.wait().expect("reap the killed mnema-server");
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

struct ServerOut {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

/// Spawn the server, write `input` (whole line-delimited JSON-RPC session) to stdin, close it
/// so EOF ends the session, and collect exit status + both streams — draining them on threads
/// so a full pipe can never deadlock the child.
fn run_server(args: &[&str], envs: &[(&str, &str)], input: &str) -> ServerOut {
    let mut child = server_cmd(args, envs).spawn().expect("spawn mnema-server");
    let mut stdin = child.stdin.take().unwrap();
    let _ = stdin.write_all(input.as_bytes());
    drop(stdin);
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let out_t = std::thread::spawn(move || drain(stdout));
    let err_t = std::thread::spawn(move || drain(stderr));
    let status = wait_with_deadline(&mut child);
    ServerOut {
        status,
        stdout: out_t.join().unwrap(),
        stderr: err_t.join().unwrap(),
    }
}

fn drain(mut stream: impl Read) -> String {
    let mut s = String::new();
    let _ = stream.read_to_string(&mut s);
    s
}

/// One line-delimited JSON-RPC request.
fn req(id: u64, method: &str, params: Value) -> String {
    format!(
        "{}\n",
        json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
    )
}

fn replies(out: &ServerOut) -> Vec<Value> {
    out.stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("every stdout line is a JSON-RPC reply"))
        .collect()
}

/// The text payload of a tools/call reply.
fn tool_text(reply: &Value) -> &str {
    reply["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("not a tool text reply: {reply}"))
}

#[test]
fn a_fresh_server_serves_prefers_path_flag_and_keeps_private_memories_hidden() {
    let td = TempDirGuard::new("fresh");
    let flag_store = td.0.join("flag.store");
    let env_store = td.0.join("env.store");
    let input = [
        req(1, "initialize", json!({})),
        req(
            2,
            "tools/call",
            json!({"name": "remember", "arguments":
                {"content": "private sentinel sk-hush", "tier": "private"}}),
        ),
        req(
            3,
            "tools/call",
            json!({"name": "remember", "arguments": {"content": "open note about rust"}}),
        ),
        req(4, "tools/call", json!({"name": "recent", "arguments": {}})),
    ]
    .concat();
    // $MNEMA_LOCAL is scrubbed entirely: an UNSET switch must leave the egress wall closed.
    let out = run_server(
        &["--path", flag_store.to_str().unwrap()],
        &[("MNEMA_PATH", env_store.to_str().unwrap())],
        &input,
    );
    assert!(out.status.success(), "stderr: {}", out.stderr);
    let r = replies(&out);
    assert_eq!(r.len(), 4, "four requests, four replies: {r:?}");
    assert_eq!(r[0]["result"]["serverInfo"]["name"], json!("mnema"));
    assert!(tool_text(&r[1]).contains("remembered as memory"), "{r:?}");
    let recent = tool_text(&r[3]);
    assert!(recent.contains("rust"), "{recent}");
    assert!(
        !recent.contains("sk-hush"),
        "an unset $MNEMA_LOCAL must keep a Private memory hidden: {recent}"
    );
    // --path beats $MNEMA_PATH: the store lands at the flag's path, not the env's.
    assert!(flag_store.exists(), "--path must decide where the store is");
    assert!(!env_store.exists(), "$MNEMA_PATH must lose to --path");
}

#[test]
fn the_egress_wall_opens_only_on_an_explicit_local_switch() {
    let td = TempDirGuard::new("wall");
    let store = td.0.join("s.store");
    let s = store.to_str().unwrap();

    // Seed the store with one Private and one Open memory through a default (Remote) server.
    let seed = [
        req(1, "initialize", json!({})),
        req(
            2,
            "tools/call",
            json!({"name": "remember", "arguments":
                {"content": "private sentinel sk-hush", "tier": "private"}}),
        ),
        req(
            3,
            "tools/call",
            json!({"name": "remember", "arguments": {"content": "open daylight note"}}),
        ),
    ]
    .concat();
    let out = run_server(&["--path", s], &[], &seed);
    assert!(out.status.success(), "seeding failed: {}", out.stderr);

    let recent = [
        req(1, "initialize", json!({})),
        req(2, "tools/call", json!({"name": "recent", "arguments": {}})),
    ]
    .concat();

    // --local (argv, a single bare flag): Private memories may surface.
    let out = run_server(&["--local"], &[("MNEMA_PATH", s)], &recent);
    assert!(out.status.success(), "stderr: {}", out.stderr);
    let text = replies(&out)
        .last()
        .map(|r| tool_text(r).to_string())
        .expect("a recent reply");
    assert!(
        text.contains("sk-hush"),
        "--local must let recall surface a Private memory: {text}"
    );

    // $MNEMA_LOCAL=1 (env): same as --local.
    let out = run_server(&[], &[("MNEMA_PATH", s), ("MNEMA_LOCAL", "1")], &recent);
    assert!(out.status.success(), "stderr: {}", out.stderr);
    let text = replies(&out)
        .last()
        .map(|r| tool_text(r).to_string())
        .expect("a recent reply");
    assert!(
        text.contains("sk-hush"),
        "$MNEMA_LOCAL=1 must open the wall: {text}"
    );

    // $MNEMA_LOCAL=0 — an operator being EXPLICIT that local mode is off — stays closed.
    let out = run_server(&[], &[("MNEMA_PATH", s), ("MNEMA_LOCAL", "0")], &recent);
    assert!(out.status.success(), "stderr: {}", out.stderr);
    let text = replies(&out)
        .last()
        .map(|r| tool_text(r).to_string())
        .expect("a recent reply");
    assert!(
        !text.contains("sk-hush"),
        "$MNEMA_LOCAL=0 must keep Private memories hidden: {text}"
    );
    assert!(text.contains("daylight"), "{text}");
}

#[test]
fn a_second_server_on_the_same_store_refuses_to_start() {
    let td = TempDirGuard::new("contend");
    let store = td.0.join("s.store");
    let s = store.to_str().unwrap();

    // Server A: spawn, prove it is up (initialize round-trips — so its lock is held), keep it
    // alive by holding its stdin open.
    let mut a = server_cmd(&["--path", s], &[])
        .spawn()
        .expect("spawn server A");
    let mut a_stdin = a.stdin.take().unwrap();
    let a_stdout = a.stdout.take().unwrap();
    a_stdin
        .write_all(req(1, "initialize", json!({})).as_bytes())
        .unwrap();
    a_stdin.flush().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut r = BufReader::new(a_stdout);
        let mut line = String::new();
        let _ = r.read_line(&mut line);
        let _ = tx.send(line);
        // Keep draining so A can never block on a full stdout pipe.
        let mut rest = String::new();
        let _ = r.read_to_string(&mut rest);
    });
    let first = match rx.recv_timeout(TIMEOUT) {
        Ok(line) => line,
        Err(_) => {
            let _ = a.kill();
            panic!("server A did not answer initialize");
        }
    };
    assert!(
        first.contains("serverInfo"),
        "server A must be serving: {first}"
    );

    // Server B on the SAME store: must refuse to start, so it can never clobber A's writes.
    let b = run_server(&["--path", s], &[], "");
    assert!(
        !b.status.success(),
        "a second server on a locked store must refuse to start (stdout: {})",
        b.stdout
    );
    assert!(
        b.stderr.contains("already in use"),
        "the refusal must say the store is in use: {}",
        b.stderr
    );

    // A exits cleanly at EOF.
    drop(a_stdin);
    let status = wait_with_deadline(&mut a);
    assert!(status.success(), "server A must exit cleanly at EOF");
    reader.join().unwrap();
}

#[test]
fn migrate_without_an_existing_store_exits_with_an_error_and_serves_nothing() {
    let td = TempDirGuard::new("migrate_missing");
    let store = td.0.join("missing.store");
    let out = run_server(
        &["--migrate", "--path", store.to_str().unwrap()],
        &[],
        &req(1, "initialize", json!({})),
    );
    assert!(
        !out.status.success(),
        "--migrate with no store must exit with an error, not serve (stdout: {})",
        out.stdout
    );
    assert!(out.stderr.contains("nothing to migrate"), "{}", out.stderr);
    assert_eq!(
        out.stdout, "",
        "a --migrate run must never answer JSON-RPC — it migrates and exits"
    );
}

#[test]
fn a_corrupt_existing_store_refuses_to_start_and_is_not_overwritten() {
    let td = TempDirGuard::new("corrupt");
    let store = td.0.join("corrupt.store");
    let garbage = b"definitely not a mnema store".as_slice();
    std::fs::write(&store, garbage).unwrap();
    let input = [
        req(1, "initialize", json!({})),
        req(
            2,
            "tools/call",
            json!({"name": "remember", "arguments": {"content": "must never land"}}),
        ),
    ]
    .concat();
    let out = run_server(&["--path", store.to_str().unwrap()], &[], &input);
    assert!(
        !out.status.success(),
        "an unopenable existing store must refuse to start (stdout: {})",
        out.stdout
    );
    assert!(
        out.stderr.contains("could not be opened"),
        "the refusal must explain itself: {}",
        out.stderr
    );
    assert_eq!(
        std::fs::read(&store).unwrap(),
        garbage,
        "refusing to start must leave the existing store byte-identical"
    );
}
