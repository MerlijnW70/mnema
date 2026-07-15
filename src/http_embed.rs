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
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectError::Request(e) => write!(f, "embeddings endpoint request failed: {e}"),
            ConnectError::BadResponse => {
                write!(f, "embeddings response had no recognizable embedding array")
            }
            ConnectError::EmptyEmbedding => write!(f, "embeddings endpoint returned width 0"),
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
    pub fn from_env() -> Result<Self, ConnectError> {
        let url = std::env::var("MNEMA_EMBED_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
        let model =
            std::env::var("MNEMA_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let api = std::env::var("MNEMA_EMBED_API")
            .ok()
            .and_then(|s| Api::parse(&s))
            .unwrap_or_else(|| Api::guess_from_url(&url));
        Self::connect(url, model, api)
    }
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
