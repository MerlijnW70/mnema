//! `HttpEmbedder` — semantic embeddings from a **local** HTTP endpoint (Ollama, llama.cpp, LM
//! Studio, text-embeddings-inference, vLLM) instead of a model bundled in-process. This gives the
//! prebuilt binary real *meaning-based* recall with **no candle / Hugging Face build weight**, and
//! lets the user pick whatever embedding model their server hosts. Behind the opt-in `http-embed`
//! feature, so the default build and the zero-dependency `mnema-core` stay untouched (umbrella-only
//! glue, like `model_embed`, below the mutation gate's behavioral waterline).
//!
//! **Local-first by intent.** The default endpoint is Ollama on loopback
//! (`http://localhost:11434/api/embeddings`); override with `$MNEMA_EMBED_URL`, pick the model with
//! `$MNEMA_EMBED_MODEL`, and select the API with `$MNEMA_EMBED_API` (`ollama` | `openai`; auto-
//! detected from a `/v1/` URL otherwise). Only plain HTTP is compiled in (no TLS): a local embedding
//! server needs none, it keeps the dependency to a few small crates, and it steers this at on-device
//! models rather than a cloud embedding API — matching mnema's privacy posture.
//!
//! Two request/response shapes are supported (see `Api`):
//! ```text
//! Ollama:  POST { "model", "prompt": <text> }  ->  { "embedding": [f32, ...] }
//! OpenAI:  POST { "model", "input":  <text> }  ->  { "data": [ { "embedding": [f32, ...] } ] }
//! ```
//! Request-build and response-parse are pure functions (`request_body`, `parse_embedding`) so
//! both shapes are unit-tested without a network.

use std::time::Duration;

use serde_json::Value;

use crate::vector::Embedder;

/// The default local embeddings endpoint (Ollama on loopback).
pub const DEFAULT_URL: &str = "http://localhost:11434/api/embeddings";
/// A small, widely-available default embedding model.
pub const DEFAULT_MODEL: &str = "nomic-embed-text";

/// Wait this long to establish a connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Wait this long for the embedding response — generous, because the first request after a cold
/// model load can be slow.
const READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Which embeddings API the endpoint speaks — they differ in request key and response shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    /// Ollama native: `POST { "model", "prompt" }` → `{ "embedding": [..] }`.
    Ollama,
    /// OpenAI-compatible (llama.cpp, LM Studio, TEI, vLLM, Ollama `/v1`):
    /// `POST { "model", "input" }` → `{ "data": [ { "embedding": [..] } ] }`.
    OpenAi,
}

impl Api {
    /// Parse an explicit selector (`"ollama"` / `"openai"`), if recognized.
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(Api::Ollama),
            "openai" | "openai-compat" | "oai" => Some(Api::OpenAi),
            _ => None,
        }
    }

    /// Guess from the URL: OpenAI-compatible endpoints conventionally live under `/v1/`.
    fn guess_from_url(url: &str) -> Self {
        if url.contains("/v1/") {
            Api::OpenAi
        } else {
            Api::Ollama
        }
    }
}

/// Why constructing an [`HttpEmbedder`] failed.
#[derive(Debug)]
pub enum ConnectError {
    /// The endpoint could not be reached, or returned a transport/HTTP error.
    Request(String),
    /// The response body held no numeric embedding for either supported API shape.
    BadResponse,
    /// The endpoint returned a zero-width embedding.
    EmptyEmbedding,
    /// `$MNEMA_EMBED_URL` points off this machine and `$MNEMA_EMBED_ALLOW_REMOTE` is not an
    /// explicit affirmative. Every `remember`/`recall` sends its FULL text — including
    /// Private-tier content — to the embeddings endpoint in plaintext HTTP, so an env var
    /// silently redirecting that stream off-device would breach "never leaks to the cloud".
    NonLocalUrl(String),
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectError::Request(e) => write!(f, "embeddings endpoint request failed: {e}"),
            ConnectError::BadResponse => {
                write!(f, "embeddings response had no recognizable embedding array")
            }
            ConnectError::EmptyEmbedding => write!(f, "embeddings endpoint returned width 0"),
            ConnectError::NonLocalUrl(url) => write!(
                f,
                "$MNEMA_EMBED_URL={url} is not a loopback address. Memory text (including \
                 private-tier content) is sent to the embeddings endpoint in plaintext, so a \
                 non-local endpoint must be opted into explicitly: set \
                 MNEMA_EMBED_ALLOW_REMOTE=1 if you really run your embedding server on \
                 another machine you trust."
            ),
        }
    }
}

impl std::error::Error for ConnectError {}

/// A text→vector [`Embedder`] backed by a local HTTP embeddings server.
pub struct HttpEmbedder {
    agent: ureq::Agent,
    url: String,
    model: String,
    api: Api,
    dims: usize,
}

impl HttpEmbedder {
    /// Connect to `url` (speaking `api`) using `model`, probing once to learn the embedding width (a
    /// store is fixed to one width, so it must be known up front). Fails if the endpoint is
    /// unreachable or returns a malformed / empty embedding.
    pub fn connect(
        url: impl Into<String>,
        model: impl Into<String>,
        api: Api,
    ) -> Result<Self, ConnectError> {
        let url = url.into();
        let model = model.into();
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout_read(READ_TIMEOUT)
            .build();
        let resp =
            request_embedding(&agent, &url, &model, api, "mnema").map_err(ConnectError::Request)?;
        let probe = parse_embedding(&resp).ok_or(ConnectError::BadResponse)?;
        if probe.is_empty() {
            return Err(ConnectError::EmptyEmbedding);
        }
        let dims = probe.len();
        Ok(Self {
            agent,
            url,
            model,
            api,
            dims,
        })
    }

    /// Connect using `$MNEMA_EMBED_URL` (default [`DEFAULT_URL`]), `$MNEMA_EMBED_MODEL` (default
    /// [`DEFAULT_MODEL`]), and `$MNEMA_EMBED_API` (`ollama`/`openai`; auto-detected from a `/v1/` URL
    /// when unset or unrecognized).
    ///
    /// **Loopback-only by default.** Every embedding request carries the memory's full text —
    /// Private tier included — as plaintext HTTP, so an env var quietly pointing off-machine
    /// would be a silent egress channel. A non-loopback `$MNEMA_EMBED_URL` is refused unless
    /// `$MNEMA_EMBED_ALLOW_REMOTE` is an explicit affirmative (`1`/`true`/`yes`/`on`). Code
    /// calling [`connect`](Self::connect) directly makes that choice explicitly and is not
    /// gated.
    pub fn from_env() -> Result<Self, ConnectError> {
        let (url, model, api) = resolve_env_config(
            std::env::var("MNEMA_EMBED_URL").ok(),
            std::env::var("MNEMA_EMBED_ALLOW_REMOTE").ok(),
            std::env::var("MNEMA_EMBED_MODEL").ok(),
            std::env::var("MNEMA_EMBED_API").ok(),
        )?;
        Self::connect(url, model, api)
    }
}

/// Resolve the [`HttpEmbedder::from_env`] configuration from the raw env-var values, pure for
/// testability: apply the defaults, enforce the loopback egress gate, and pick the API.
/// `from_env` is a thin shell that reads `$MNEMA_EMBED_URL` / `$MNEMA_EMBED_ALLOW_REMOTE` /
/// `$MNEMA_EMBED_MODEL` / `$MNEMA_EMBED_API` and delegates here.
fn resolve_env_config(
    url: Option<String>,
    allow_remote: Option<String>,
    model: Option<String>,
    api: Option<String>,
) -> Result<(String, String, Api), ConnectError> {
    let url = url.unwrap_or_else(|| DEFAULT_URL.to_string());
    if !remote_url_permitted(&url, allow_remote.as_deref()) {
        return Err(ConnectError::NonLocalUrl(url));
    }
    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let api = api
        .and_then(|s| Api::parse(&s))
        .unwrap_or_else(|| Api::guess_from_url(&url));
    Ok((url, model, api))
}

/// Whether `url` targets this machine: a `localhost`, `127.x.y.z`, or `[::1]` host. Only plain
/// `http://` is compiled in, so no other scheme needs parsing.
fn is_loopback_url(url: &str) -> bool {
    let rest = url.strip_prefix("http://").unwrap_or(url);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // Drop any userinfo, then the port — IPv6 hosts are bracketed, so `[::1]:8080` splits on `]`.
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => host.split(':').next().unwrap_or(""),
    };
    let h = host.to_ascii_lowercase();
    h == "localhost" || h == "::1" || h.starts_with("127.")
}

/// The `from_env` egress gate, pure for testability: a loopback `url` is always permitted; a
/// non-loopback one only when `allow` (the `$MNEMA_EMBED_ALLOW_REMOTE` value) is an explicit
/// affirmative. Fail-closed like every switch guarding the privacy posture: `0`, `false`, an
/// empty string, or anything unrecognized does NOT open the wall.
fn remote_url_permitted(url: &str, allow: Option<&str>) -> bool {
    is_loopback_url(url)
        || matches!(
            allow.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
            Some("1" | "true" | "yes" | "on")
        )
}

/// The request body for `api`.
fn request_body(api: Api, model: &str, text: &str) -> Value {
    match api {
        Api::Ollama => serde_json::json!({ "model": model, "prompt": text }),
        Api::OpenAi => serde_json::json!({ "model": model, "input": text }),
    }
}

/// Extract the embedding vector from a response body, accepting either the Ollama shape
/// (`{ "embedding": [..] }`) or the OpenAI-compatible shape (`{ "data": [ { "embedding": [..] } ] }`).
fn parse_embedding(resp: &Value) -> Option<Vec<f32>> {
    let arr = resp
        .get("embedding")
        .and_then(Value::as_array)
        .or_else(|| {
            resp.get("data")?
                .as_array()?
                .first()?
                .get("embedding")?
                .as_array()
        })?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        out.push(x.as_f64()? as f32);
    }
    Some(out)
}

/// POST one embedding request (plain HTTP) and return the parsed response body. The JSON body is
/// serialized with `serde_json` and sent as a string, so `ureq` needs neither its `json` nor `tls`
/// features — keeping the opt-in dependency minimal.
fn request_embedding(
    agent: &ureq::Agent,
    url: &str,
    model: &str,
    api: Api,
    text: &str,
) -> Result<Value, String> {
    let body = request_body(api, model, text).to_string();
    let resp = agent
        .post(url)
        .set("Content-Type", "application/json")
        .send_string(&body)
        .map_err(|e| e.to_string())?;
    let text = resp.into_string().map_err(|e| e.to_string())?;
    serde_json::from_str::<Value>(&text).map_err(|e| e.to_string())
}

impl Embedder for HttpEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let parsed = match request_embedding(&self.agent, &self.url, &self.model, self.api, text) {
            Ok(resp) => parse_embedding(&resp),
            Err(_) => None,
        };
        match parsed {
            Some(v) if v.len() == self.dims => v,
            // Degrade like the other embedders: a zero vector of the fixed width keeps the index
            // consistent (the memory is stored, just unmatched this call) rather than panicking
            // mid-recall on a transient endpoint failure or an unexpected width.
            Some(v) => {
                eprintln!(
                    "mnema: HttpEmbedder got width {} (expected {}); returning zero vector",
                    v.len(),
                    self.dims
                );
                vec![0.0; self.dims]
            }
            None => {
                eprintln!(
                    "mnema: HttpEmbedder request to {} failed; returning zero vector",
                    self.url
                );
                vec![0.0; self.dims]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;

    /// Serve `bodies.len()` sequential HTTP requests on a fresh loopback listener, answering the
    /// i-th request with the i-th canned JSON body, then exit. Returns the endpoint URL and the
    /// server thread's join handle. Every response says `Connection: close`, so the client opens a
    /// new connection per request and the accept loop stays strictly sequential. The read side is
    /// timeout-guarded so a wedged client cannot hang the suite.
    fn spawn_canned_server(bodies: Vec<String>) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback listener");
        let port = listener.local_addr().expect("local addr").port();
        let handle = std::thread::spawn(move || {
            for body in bodies {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
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
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        (format!("http://127.0.0.1:{port}/api/embeddings"), handle)
    }

    #[test]
    fn connect_learns_the_width_from_a_nonempty_probe() {
        let (url, server) = spawn_canned_server(vec![r#"{"embedding":[1.0,2.0,3.0]}"#.into()]);
        let embedder = HttpEmbedder::connect(url.as_str(), "m", Api::Ollama)
            .expect("a non-empty probe embedding must connect");
        assert_eq!(embedder.dims(), 3);
        server.join().expect("server thread");
    }

    #[test]
    fn connect_refuses_a_zero_width_probe() {
        let (url, server) = spawn_canned_server(vec![r#"{"embedding":[]}"#.into()]);
        match HttpEmbedder::connect(url.as_str(), "m", Api::Ollama) {
            Err(ConnectError::EmptyEmbedding) => {}
            Err(e) => panic!("expected EmptyEmbedding, got {e:?}"),
            Ok(_) => panic!("a width-0 embedding must not connect"),
        }
        server.join().expect("server thread");
    }

    #[test]
    fn embed_returns_the_vector_on_a_width_match_and_zeros_on_a_mismatch() {
        let (url, server) = spawn_canned_server(vec![
            r#"{"embedding":[1.0,2.0,3.0]}"#.into(), // connect probe -> dims = 3
            r#"{"embedding":[0.5,1.5,2.5]}"#.into(), // matching width -> passed through
            r#"{"embedding":[9.0,9.0]}"#.into(),     // wrong width -> zero vector of dims
        ]);
        let embedder = HttpEmbedder::connect(url.as_str(), "m", Api::Ollama).expect("connect");
        assert_eq!(embedder.embed("a"), vec![0.5_f32, 1.5, 2.5]);
        assert_eq!(embedder.embed("b"), vec![0.0_f32, 0.0, 0.0]);
        server.join().expect("server thread");
    }

    #[test]
    fn env_config_defaults_gates_remote_urls_and_picks_the_api() {
        // No env vars: the loopback default is permitted, model defaults, API guessed as Ollama.
        let (url, model, api) =
            resolve_env_config(None, None, None, None).expect("defaults are loopback");
        assert_eq!(url, DEFAULT_URL);
        assert_eq!(model, DEFAULT_MODEL);
        assert_eq!(api, Api::Ollama);

        // A remote URL without the explicit opt-in is refused (the egress wall)...
        let remote = "http://192.168.1.20:11434/api/embeddings";
        match resolve_env_config(Some(remote.into()), None, None, None) {
            Err(ConnectError::NonLocalUrl(u)) => assert_eq!(u, remote),
            other => panic!("a remote URL without opt-in must be refused, got {other:?}"),
        }
        // ...and passed through with an explicit affirmative.
        let (url, model, api) = resolve_env_config(
            Some(remote.into()),
            Some("1".into()),
            Some("mm".into()),
            None,
        )
        .expect("explicit opt-in");
        assert_eq!(url, remote);
        assert_eq!(model, "mm");
        assert_eq!(api, Api::Ollama);

        // The API comes from the selector when recognized, else is guessed from the URL.
        let v1 = "http://127.0.0.1:8080/v1/embeddings";
        let (_, _, api) = resolve_env_config(Some(v1.into()), None, None, None).expect("loopback");
        assert_eq!(api, Api::OpenAi);
        let (_, _, api) = resolve_env_config(Some(v1.into()), None, None, Some("ollama".into()))
            .expect("loopback");
        assert_eq!(api, Api::Ollama);
    }

    #[test]
    fn request_body_matches_each_api_shape() {
        let ollama = request_body(Api::Ollama, "m", "hello");
        assert_eq!(ollama["model"], "m");
        assert_eq!(ollama["prompt"], "hello");
        assert!(ollama.get("input").is_none());

        let openai = request_body(Api::OpenAi, "m", "hello");
        assert_eq!(openai["model"], "m");
        assert_eq!(openai["input"], "hello");
        assert!(openai.get("prompt").is_none());
    }

    #[test]
    fn parse_embedding_accepts_both_shapes() {
        // Ollama: top-level "embedding".
        let ollama = serde_json::json!({ "embedding": [0.25, -0.5, 1.0] });
        assert_eq!(parse_embedding(&ollama), Some(vec![0.25_f32, -0.5, 1.0]));
        // OpenAI: nested under data[0].embedding.
        let openai = serde_json::json!({ "data": [ { "embedding": [1.0, 2.0] } ] });
        assert_eq!(parse_embedding(&openai), Some(vec![1.0_f32, 2.0]));
    }

    #[test]
    fn parse_embedding_rejects_missing_or_non_numeric() {
        assert_eq!(parse_embedding(&serde_json::json!({})), None);
        assert_eq!(
            parse_embedding(&serde_json::json!({ "embedding": "no" })),
            None
        );
        assert_eq!(
            parse_embedding(&serde_json::json!({ "embedding": [1.0, "x"] })),
            None
        );
        assert_eq!(parse_embedding(&serde_json::json!({ "data": [] })), None);
    }

    #[test]
    fn loopback_detection_covers_the_local_spellings_and_rejects_the_rest() {
        for local in [
            "http://localhost:11434/api/embeddings",
            "http://LOCALHOST/v1/embeddings",
            "http://127.0.0.1:8080/v1/embeddings",
            "http://127.5.5.5/api/embeddings",
            "http://[::1]:11434/api/embeddings",
        ] {
            assert!(is_loopback_url(local), "{local} is this machine");
        }
        for remote in [
            "http://192.168.1.20:11434/api/embeddings",
            "http://embed.example.com/v1/embeddings",
            "http://10.0.0.1/api/embeddings",
            "http://[fe80::1]:11434/api/embeddings",
            // Tricks: loopback as userinfo/port text, not as the host.
            "http://evil.example.com:11434/api?host=localhost",
            "http://localhost.example.com/api/embeddings",
        ] {
            assert!(!is_loopback_url(remote), "{remote} is NOT this machine");
        }
    }

    #[test]
    fn a_remote_embed_url_needs_an_explicit_affirmative_opt_in() {
        let remote = "http://192.168.1.20:11434/api/embeddings";
        // Loopback needs no opt-in.
        assert!(remote_url_permitted(DEFAULT_URL, None));
        // Remote without the env var, or with a falsy/unrecognized value, is REFUSED —
        // private-tier text would flow there in plaintext.
        for no in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("off"),
            Some("remote"),
        ] {
            assert!(
                !remote_url_permitted(remote, no),
                "{no:?} must not open the egress wall"
            );
        }
        for yes in [Some("1"), Some("true"), Some("YES"), Some(" on ")] {
            assert!(
                remote_url_permitted(remote, yes),
                "{yes:?} is an explicit opt-in"
            );
        }
    }

    #[test]
    fn api_selection_parses_and_guesses() {
        assert_eq!(Api::parse("openai"), Some(Api::OpenAi));
        assert_eq!(Api::parse("Ollama"), Some(Api::Ollama));
        assert_eq!(Api::parse("nonsense"), None);
        assert_eq!(
            Api::guess_from_url("http://x:8080/v1/embeddings"),
            Api::OpenAi
        );
        assert_eq!(
            Api::guess_from_url("http://localhost:11434/api/embeddings"),
            Api::Ollama
        );
    }
}
