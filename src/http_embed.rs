//! [`HttpEmbedder`] — semantic embeddings from a **local** HTTP endpoint (Ollama, llama.cpp, LM
//! Studio, text-embeddings-inference) instead of a model bundled in-process. This gives the prebuilt
//! binary real *meaning-based* recall with **no candle / Hugging Face build weight**, and lets the
//! user pick whatever embedding model their server hosts. Behind the opt-in `http-embed` feature, so
//! the default build and the zero-dependency `mnema-core` stay untouched (this is umbrella-only glue,
//! exactly like `model_embed` — it sits below the mutation gate's behavioral waterline).
//!
//! **Local-first by intent.** The default endpoint is Ollama on loopback
//! (`http://localhost:11434/api/embeddings`); override with `$MNEMA_EMBED_URL` and pick the model
//! with `$MNEMA_EMBED_MODEL`. Only plain HTTP is compiled in (no TLS): a local embedding server
//! needs none, it keeps the dependency to a few small crates, and it steers this at on-device models
//! rather than a cloud embedding API — matching mnema's privacy posture.
//!
//! Wire format follows Ollama's native embeddings API:
//! ```text
//! POST { "model": <model>, "prompt": <text> }   ->   { "embedding": [f32, ...] }
//! ```
//! The request-build and response-parse steps are pure functions ([`request_body`],
//! [`parse_embedding`]) so they are unit-tested without a network.

use serde_json::Value;

use crate::vector::Embedder;

/// The default local embeddings endpoint (Ollama on loopback).
pub const DEFAULT_URL: &str = "http://localhost:11434/api/embeddings";
/// A small, widely-available default embedding model.
pub const DEFAULT_MODEL: &str = "nomic-embed-text";

/// Why constructing an [`HttpEmbedder`] failed.
#[derive(Debug)]
pub enum ConnectError {
    /// The endpoint could not be reached, or returned a transport/HTTP error.
    Request(String),
    /// The response body had no numeric `"embedding"` array (wrong endpoint or API shape).
    BadResponse,
    /// The endpoint returned a zero-width embedding.
    EmptyEmbedding,
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectError::Request(e) => write!(f, "embeddings endpoint request failed: {e}"),
            ConnectError::BadResponse => {
                write!(f, "embeddings response had no numeric \"embedding\" array")
            }
            ConnectError::EmptyEmbedding => write!(f, "embeddings endpoint returned width 0"),
        }
    }
}

impl std::error::Error for ConnectError {}

/// A text→vector [`Embedder`] backed by a local HTTP embeddings server.
pub struct HttpEmbedder {
    url: String,
    model: String,
    dims: usize,
}

impl HttpEmbedder {
    /// Connect to `url` using `model`, probing once to learn the embedding width (a store is fixed to
    /// one width, so this must be known up front). Fails if the endpoint is unreachable or returns a
    /// malformed / empty embedding.
    pub fn connect(url: impl Into<String>, model: impl Into<String>) -> Result<Self, ConnectError> {
        let url = url.into();
        let model = model.into();
        let resp = request_embedding(&url, &model, "mnema").map_err(ConnectError::Request)?;
        let probe = parse_embedding(&resp).ok_or(ConnectError::BadResponse)?;
        if probe.is_empty() {
            return Err(ConnectError::EmptyEmbedding);
        }
        let dims = probe.len();
        Ok(Self { url, model, dims })
    }

    /// Connect using `$MNEMA_EMBED_URL` (default [`DEFAULT_URL`]) and `$MNEMA_EMBED_MODEL`
    /// (default [`DEFAULT_MODEL`]).
    pub fn from_env() -> Result<Self, ConnectError> {
        let url = std::env::var("MNEMA_EMBED_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
        let model =
            std::env::var("MNEMA_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        Self::connect(url, model)
    }
}

/// The request body Ollama's `/api/embeddings` expects.
fn request_body(model: &str, text: &str) -> Value {
    serde_json::json!({ "model": model, "prompt": text })
}

/// Extract the embedding vector from a response body, if it holds a numeric `"embedding"` array.
fn parse_embedding(resp: &Value) -> Option<Vec<f32>> {
    let arr = resp.get("embedding")?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        out.push(x.as_f64()? as f32);
    }
    Some(out)
}

/// POST one embedding request (plain HTTP) and return the parsed response body. The JSON body is
/// serialized with `serde_json` and sent as a string, so `ureq` needs neither its `json` nor `tls`
/// features — keeping the opt-in dependency minimal.
fn request_embedding(url: &str, model: &str, text: &str) -> Result<Value, String> {
    let body = request_body(model, text).to_string();
    let resp = ureq::post(url)
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
        let parsed = match request_embedding(&self.url, &self.model, text) {
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
    fn request_body_matches_the_ollama_shape() {
        let b = request_body("nomic-embed-text", "hello world");
        assert_eq!(b["model"], "nomic-embed-text");
        assert_eq!(b["prompt"], "hello world");
    }

    #[test]
    fn parse_embedding_reads_a_numeric_vector() {
        let resp = serde_json::json!({ "embedding": [0.25, -0.5, 1.0] });
        assert_eq!(parse_embedding(&resp), Some(vec![0.25_f32, -0.5, 1.0]));
    }

    #[test]
    fn parse_embedding_rejects_missing_or_non_numeric() {
        assert_eq!(parse_embedding(&serde_json::json!({})), None); // no field
        assert_eq!(
            parse_embedding(&serde_json::json!({ "embedding": "no" })),
            None
        ); // not an array
        assert_eq!(
            parse_embedding(&serde_json::json!({ "embedding": [1.0, "x"] })),
            None // a non-numeric element voids the whole vector
        );
    }
}
