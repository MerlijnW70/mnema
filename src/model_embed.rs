//! A real, in-process **semantic** embedder — sentence-transformers `all-MiniLM-L6-v2` run
//! through [candle](https://github.com/huggingface/candle) (pure Rust: no Python, no ONNX C++
//! runtime). Behind the opt-in `local-embed` feature so the zero-dependency core is untouched.
//!
//! This is I/O + model glue that sits **below noha's behavioral waterline** (like the CLI): it
//! is not in the probed `sources`, because a transformer forward pass is not deterministic
//! logic a mutation ratchet can pin. The *pooling math* it rests on — [`mean_pool`] and
//! [`l2_normalize`] — lives in [`crate::embed`] and **is** probed, so the load-bearing
//! arithmetic (which token vectors are averaged, unit-normalisation) stays gate-covered while
//! candle owns the matmuls.
//!
//! `#![forbid(unsafe_code)]` (crate-wide) still holds here: weights are loaded via candle's
//! **safe** `from_slice_safetensors`, not the `unsafe` mmap path the upstream example uses.

use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use crate::embed::{l2_normalize, mean_pool};
use crate::vector::Embedder;

/// The Hugging Face model id. all-MiniLM-L6-v2 is a 6-layer MiniLM producing 384-dim
/// sentence embeddings — the de-facto default small local embedding model.
const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// The model's output width. Fixed by the checkpoint; also the width the vector index uses.
const DIMS: usize = 384;

/// Anything that can go wrong loading the model (network, missing asset, malformed weights).
#[derive(Debug)]
pub struct LoadError(String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MiniLmEmbedder load failed: {}", self.0)
    }
}
impl std::error::Error for LoadError {}

fn err<E: std::fmt::Display>(e: E) -> LoadError {
    LoadError(e.to_string())
}

/// A local semantic embedder over all-MiniLM-L6-v2. Construct once (it downloads/loads the
/// model), then reuse — `embed` is `&self`. CPU inference; deterministic for a given input.
pub struct MiniLmEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: candle_core::Device,
}

impl MiniLmEmbedder {
    /// Load all-MiniLM-L6-v2, fetching `config.json`, `tokenizer.json`, and `model.safetensors`
    /// from Hugging Face on first use and caching them (later loads are offline from the HF
    /// cache). Runs on CPU.
    pub fn load() -> Result<Self, LoadError> {
        let device = candle_core::Device::Cpu;
        let repo = Api::new()
            .map_err(err)?
            .repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

        let config_path = repo.get("config.json").map_err(err)?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(err)?;
        let weights_path = repo.get("model.safetensors").map_err(err)?;

        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(config_path).map_err(err)?)
                .map_err(err)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(err)?;

        // SAFE load (read bytes → from_slice_safetensors), not the unsafe mmap variant — the
        // crate forbids unsafe. Costs one full read of the weights; fine for a one-time load.
        let weights = std::fs::read(weights_path).map_err(err)?;
        let vb =
            VarBuilder::from_slice_safetensors(&weights, DTYPE, &device).map_err(err)?;
        let model = BertModel::load(vb, &config).map_err(err)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Embed one text, propagating candle/tokenizer errors instead of swallowing them — used by
    /// the `Embedder` impl, which degrades an error to the zero vector.
    fn try_embed(&self, text: &str) -> Result<Vec<f32>, candle_core::Error> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(candle_core::Error::wrap)?;
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();

        // [1, seq_len] batch of one.
        let token_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::new(mask, &self.device)?.unsqueeze(0)?;

        // [1, seq_len, 384] → [seq_len, 384] → Vec per token.
        let hidden = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?
            .squeeze(0)?;
        let token_embeddings: Vec<Vec<f32>> = hidden.to_vec2::<f32>()?;

        // Pool + normalise in the crate's own TESTED math (mask drives which tokens count).
        Ok(l2_normalize(mean_pool(&token_embeddings, mask)))
    }
}

impl Embedder for MiniLmEmbedder {
    fn dims(&self) -> usize {
        DIMS
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        match self.try_embed(text) {
            Ok(v) => v,
            Err(e) => {
                // Degrade like a broken embedder elsewhere in the crate: a zero vector keeps
                // the index width consistent (so the memory is stored, just unmatched) rather
                // than panicking mid-recall.
                eprintln!("engram: MiniLmEmbedder inference failed ({e}); returning zero vector");
                vec![0.0; DIMS]
            }
        }
    }
}
