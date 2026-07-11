//! Public recall benchmark on **LoCoMo** (Maharana et al., long-conversation memory) — the
//! retrieval task mnemo and others report on, run deterministically (no LLM judge): ingest every
//! conversation turn as a memory, then for each question measure whether the gold **evidence**
//! turns land in engram's top-k. This is a *leaderboard-comparable* recall number, unlike the
//! in-repo `recall` bench (which is engram's own paraphrase fixture).
//!
//! The dataset is not vendored (it's a third party's, ~2.8 MB). Download it and point at it:
//! ```bash
//! curl -sL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o locomo10.json
//! export LOCOMO_PATH=$PWD/locomo10.json
//! bash scripts/locomo.sh          # robust full run (recommended)
//! ```
//!
//! **Full run → `scripts/locomo.sh`.** candle accumulates native memory over thousands of forward
//! passes and can flakily crash a single long process; the driver runs **one conversation per
//! fresh subprocess** (via `LOCOMO_ONLY=i`, printing machine-readable `SEM <r5> <r10> <n>`) with
//! retry, then aggregates — each conversation is isolated, so the buildup can't reach across them.
//! Running this bench directly does a quick lexical pass plus a best-effort in-process semantic
//! pass (fine for a few conversations; use the driver for all 10).
//!
//! Reported: mean **Recall@k** = (gold evidence turns retrieved in top-k) / (gold evidence turns),
//! averaged over every answerable question, for the lexical default vs the semantic
//! (all-MiniLM-L6-v2 + dense-weighted) path. Full-dataset result (2026-07): lexical R@5 0.225 /
//! R@10 0.323 · semantic R@5 0.385 / R@10 0.453 (+16.0 pts R@5 over 1981 questions).

use engram::facade::Engram;
use engram::retrieval::RetrievalWeights;
use engram::vector::Embedder;
use engram::{Destination, EgressTier};
use std::collections::HashMap;

/// Pull the dia_ids (`D<session>:<turn>`) out of LoCoMo's `evidence` field, which is a *stringified*
/// Python list like `"['D1:3', 'D2:5']"` — scan for the `D…:…` token shape rather than JSON-parsing it.
fn evidence_ids(raw: &str) -> Vec<String> {
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == ':' { c } else { ' ' })
        .collect();
    cleaned
        .split_whitespace()
        .filter(|t| {
            let b = t.as_bytes();
            b.first() == Some(&b'D') && t.contains(':') && t[1..].chars().next().is_some_and(|c| c.is_ascii_digit())
        })
        .map(str::to_string)
        .collect()
}

/// Summed Recall@5 / Recall@10 (and question count) over conversation samples `start..end`,
/// computed in a SINGLE ingest pass (retrieve top-10 once per question; count gold hits within the
/// 5 and 10 cutoffs). Returns *sums* so a batched caller can aggregate — the semantic path
/// processes the dataset in batches, reloading the model between them to release the native memory
/// candle accumulates over thousands of forward passes.
fn eval_range<E: Embedder>(data: &serde_json::Value, make: impl Fn() -> E, weights: RetrievalWeights, start: usize, end: usize) -> (f64, f64, usize) {
    let mut r5_sum = 0.0;
    let mut r10_sum = 0.0;
    let mut questions = 0usize;

    let all = data.as_array().expect("dataset is an array");
    for sample in &all[start..end.min(all.len())] {
        let conv = &sample["conversation"];
        let mut mem = Engram::new(make());
        let mut id_of: HashMap<String, u64> = HashMap::new();

        // Ingest every turn of every session as a memory, keyed by its dia_id.
        if let Some(obj) = conv.as_object() {
            for (key, val) in obj {
                // session_N is an array of turns; session_N_date_time and speaker_* are not.
                if !(key.starts_with("session_") && val.is_array()) {
                    continue;
                }
                for turn in val.as_array().unwrap() {
                    let (Some(dia), Some(text)) = (turn["dia_id"].as_str(), turn["text"].as_str()) else {
                        continue;
                    };
                    let speaker = turn["speaker"].as_str().unwrap_or("");
                    let id = mem.remember(EgressTier::Open, &format!("{speaker}: {text}"));
                    id_of.insert(dia.to_string(), id);
                }
            }
        }

        for qa in sample["qa"].as_array().into_iter().flatten() {
            let Some(question) = qa["question"].as_str() else { continue };
            let gold_raw = match &qa["evidence"] {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            // Gold evidence turns we actually stored (adversarial/no-evidence questions drop out).
            let gold: Vec<u64> = evidence_ids(&gold_raw)
                .iter()
                .filter_map(|d| id_of.get(d).copied())
                .collect();
            if gold.is_empty() {
                continue;
            }

            // Retrieve the top 10 once (per_retriever wide, budget huge so nothing is truncated).
            let hits = mem.recall_weighted(question, Destination::Local, 20, 1_000_000, weights);
            let ids: Vec<u64> = hits.iter().take(10).map(|b| b.id).collect();
            let denom = gold.len() as f64;
            let in5 = gold.iter().filter(|g| ids.iter().take(5).any(|i| i == *g)).count();
            let in10 = gold.iter().filter(|g| ids.contains(g)).count();
            r5_sum += in5 as f64 / denom;
            r10_sum += in10 as f64 / denom;
            questions += 1;
        }
    }
    (r5_sum, r10_sum, questions)
}

#[cfg(all(feature = "secure", feature = "local-embed"))]
fn main() {
    use engram::embed::HashEmbedder;
    use engram::model_embed::MiniLmEmbedder;

    let path = std::env::var("LOCOMO_PATH")
        .expect("set LOCOMO_PATH to a downloaded data/locomo10.json (see the bench header)");
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read LOCOMO_PATH")).expect("parse LoCoMo json");

    let total = data.as_array().map_or(0, |a| a.len());

    // Single-conversation mode: process exactly one sample's SEMANTIC eval in this (fresh) process
    // and print machine-readable sums. A shell loop runs one process per conversation and
    // aggregates — each is isolated, so the candle native-memory buildup on long runs can't reach
    // across conversations, and a flaky crash retries just that one.
    if let Ok(i) = std::env::var("LOCOMO_ONLY").map(|v| v.parse::<usize>().unwrap()) {
        let model = std::sync::Arc::new(MiniLmEmbedder::load().expect("load all-MiniLM-L6-v2"));
        let (r5s, r10s, n) = eval_range(&data, || ArcEmbedder(model.clone()), RetrievalWeights::semantic(), i, i + 1);
        println!("SEM {r5s} {r10s} {n}");
        return;
    }

    println!("LoCoMo retrieval — mean Recall@k over all answerable questions ({total} conversations)\n");
    println!("  config      R@5      R@10     n");

    // Lexical: pure Rust, no accumulation — one pass over the whole dataset.
    let (l5s, l10s, n) = eval_range(&data, || HashEmbedder::new(HashEmbedder::DEFAULT_DIMS), RetrievalWeights::default(), 0, total);
    let (l5, l10) = (l5s / n as f64, l10s / n as f64);
    println!("  lexical     {l5:.3}    {l10:.3}    {n}");

    // Semantic: batch the dataset and reload the model per batch (candle accumulates native memory
    // over thousands of forwards; a fresh model between batches releases it). Batch of 4 is well
    // inside the safe range (5 samples ran clean).
    let (mut s5s, mut s10s, mut sn) = (0.0, 0.0, 0usize);
    let mut start = 0;
    while start < total {
        let end = (start + 4).min(total);
        eprintln!("  [semantic batch {start}..{end}] loading model…");
        let model = std::sync::Arc::new(MiniLmEmbedder::load().expect("load all-MiniLM-L6-v2"));
        let (a, b, c) = eval_range(&data, || ArcEmbedder(model.clone()), RetrievalWeights::semantic(), start, end);
        s5s += a;
        s10s += b;
        sn += c;
        start = end;
    }
    let (s5, s10) = (s5s / sn as f64, s10s / sn as f64);
    println!("  semantic    {s5:.3}    {s10:.3}    {sn}");

    println!("\n  R@5 lift from the semantic path: {:+.1} percentage points", (s5 - l5) * 100.0);
}

/// A cheap `Embedder` handle so one loaded model is shared across every conversation's `Engram`
/// (each `evaluate` call builds a fresh store per sample). Only needed by the semantic path.
#[cfg(all(feature = "secure", feature = "local-embed"))]
struct ArcEmbedder(std::sync::Arc<engram::model_embed::MiniLmEmbedder>);
#[cfg(all(feature = "secure", feature = "local-embed"))]
impl Embedder for ArcEmbedder {
    fn dims(&self) -> usize {
        self.0.dims()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        self.0.embed(text)
    }
}

#[cfg(not(all(feature = "secure", feature = "local-embed")))]
fn main() {
    println!("run with `--features secure,local-embed` and LOCOMO_PATH set (see the bench header).");
}
