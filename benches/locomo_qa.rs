//! **End-to-end QA-accuracy** on LoCoMo — the headline metric memory systems (Mem0, Zep, LangMem)
//! report, run **fully locally** with no API key. Where `benches/locomo.rs` measures *retrieval*
//! (do the gold evidence turns land in top-k), this measures whether an LLM, given **only** the
//! memories mnema retrieves, actually **answers the question correctly** — the number that reflects
//! the whole pipeline (retrieval + the grounding it enables), judged by an LLM.
//!
//! Local-first, on brand: both the answerer and the judge are a local model served by Ollama
//! (native `/api/generate`). Nothing leaves the machine — no API key, no cloud.
//!
//! ```bash
//! ollama serve &                 # a local LLM runtime
//! ollama pull llama3.2:3b        # the answerer + judge (override with $QA_MODEL)
//! curl -sL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o locomo10.json
//! LOCOMO_PATH=$PWD/locomo10.json \
//!   cargo bench --bench locomo_qa --features secure,local-embed,http-embed
//! ```
//!
//! ## Method (LLM-as-judge, no gold-string matching)
//! For each sampled answerable question: ingest the conversation into mnema, retrieve the top-k
//! memories with the **semantic** path (all-MiniLM-L6-v2 + dense-weighted fusion), ask the LLM to
//! answer using *only* those memories, then ask the LLM to grade that answer against the dataset's
//! reference answer (lenient factual match — "7 May 2023" == "May 7th 2023"). Accuracy = fraction
//! graded CORRECT. Category 5 (adversarial / unanswerable) is skipped — it measures abstention, not
//! grounded QA. Temperature 0 throughout for reproducibility.
//!
//! ## Scope
//! Runs a bounded **sample** (default 50 questions, spread across the first `$QA_CONVS` conversations)
//! so a CPU-only local model finishes in minutes — the count `n` is always printed. Raise `$QA_SAMPLE`
//! / `$QA_CONVS` for a fuller run. Config: `$QA_MODEL`, `$QA_URL`, `$QA_SAMPLE`, `$QA_TOPK`, `$QA_CONVS`.
//!
//! ## Observed baseline (read the caveat)
//! 2026-07-15, 25-question sample (5 per conversation × 5), top-10, **llama3.2:3b as both answerer
//! and judge**: overall **0.16** (cat1 0.30, cat2 0.00, cat3 0.00, cat4 0.50). This is
//! **answerer-bound, not a retrieval ceiling** — the retrieval is working (the retrieval-recall
//! number is in `benches/locomo.rs`: semantic R@5 0.40). The misses are dominated by the small
//! model: on temporal questions it returns *near-correct* dates ("20 January" for gold "19 January",
//! "22 December" for "21 December") — proof the evidence was retrieved, but its relative-date
//! arithmetic ("yesterday" = the session date − 1) is wrong — and a strict 3B judge rejects
//! partially-correct answers ("Taekwondo" vs "Kickboxing, Taekwondo"). Point `$QA_MODEL` at a
//! stronger endpoint (a bigger local model, or your own) to raise it; this is a floor, not a claim.

#[cfg(all(feature = "secure", feature = "local-embed", feature = "http-embed"))]
fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One `/api/generate` call to the local Ollama endpoint at temperature 0. Returns the response
/// text, or an empty string on any transport / parse failure (the caller treats that as a wrong
/// answer, so a flaky call can never inflate accuracy). JSON is built + parsed with `serde_json`
/// and sent as a string, so `ureq` needs neither its `json` nor `tls` feature (matches `http_embed`).
#[cfg(all(feature = "secure", feature = "local-embed", feature = "http-embed"))]
fn generate(agent: &ureq::Agent, url: &str, model: &str, prompt: &str) -> String {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": { "temperature": 0.0 },
    })
    .to_string();
    let resp = agent
        .post(&format!("{url}/api/generate"))
        .set("Content-Type", "application/json")
        .send_string(&body);
    // Nested matches (not `?`/`and_then`) so the large `ureq::Error` never rides in a Result — keeps
    // clippy's `result_large_err` quiet — and any failure degrades to "" (a wrong answer).
    let body_text = match resp {
        Ok(r) => match r.into_string() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  llm read failed: {e}");
                return String::new();
            }
        },
        Err(e) => {
            eprintln!("  llm call failed: {e}");
            return String::new();
        }
    };
    serde_json::from_str::<serde_json::Value>(&body_text)
        .ok()
        .and_then(|v| {
            v.get("response")
                .and_then(serde_json::Value::as_str)
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_default()
}

/// Grade a judge verdict robustly: `INCORRECT` contains `CORRECT`, so check the negative first.
#[cfg(all(feature = "secure", feature = "local-embed", feature = "http-embed"))]
fn verdict_is_correct(raw: &str) -> bool {
    let up = raw.to_ascii_uppercase();
    if up.contains("INCORRECT") {
        false
    } else {
        up.contains("CORRECT")
    }
}

#[cfg(all(feature = "secure", feature = "local-embed", feature = "http-embed"))]
fn main() {
    use mnema::facade::Mnema;
    use mnema::model_embed::MiniLmEmbedder;
    use mnema::retrieval::RetrievalWeights;
    use mnema::{Destination, EgressTier};
    use std::collections::BTreeMap;
    use std::time::Duration;

    let path = std::env::var("LOCOMO_PATH")
        .expect("set LOCOMO_PATH to a downloaded data/locomo10.json (see the bench header)");
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read LOCOMO_PATH"))
            .expect("parse LoCoMo json");
    let all = data.as_array().expect("dataset is an array");

    let url = std::env::var("QA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("QA_MODEL").unwrap_or_else(|_| "llama3.2:3b".to_string());
    let sample: usize = env_or("QA_SAMPLE", 50);
    let topk: usize = env_or("QA_TOPK", 15);
    let n_convs: usize = env_or::<usize>("QA_CONVS", 5).min(all.len());
    let per_conv = sample.div_ceil(n_convs);

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(180))
        .build();

    println!(
        "LoCoMo end-to-end QA accuracy — semantic retrieval (top-{topk}) + {model} answerer/judge, \
         local via Ollama\n  sampling up to {per_conv} answerable questions from each of {n_convs} \
         conversations (target {sample})\n"
    );

    let mut correct = 0usize;
    let mut total = 0usize;
    // category -> (correct, total)
    let mut by_cat: BTreeMap<u64, (usize, usize)> = BTreeMap::new();

    for sample_conv in all.iter().take(n_convs) {
        let conv = &sample_conv["conversation"];
        let mut mem = Mnema::new(MiniLmEmbedder::load().expect("load all-MiniLM-L6-v2"));

        // Ingest every turn of every session as a memory, prefixed with the session's date. LoCoMo
        // turns speak in RELATIVE time ("yesterday", "last year"); the absolute date lives in the
        // sibling `session_N_date_time` field, so a memory without it can't answer a "when" question
        // (category 2). Stamping each memory with its date is also just how a real memory layer
        // stores things — timestamped — so this reflects mnema's intended use, not dataset-fitting.
        if let Some(obj) = conv.as_object() {
            for (key, val) in obj {
                if !(key.starts_with("session_") && val.is_array()) {
                    continue;
                }
                let date = obj
                    .get(&format!("{key}_date_time"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                for turn in val.as_array().unwrap() {
                    let (Some(_dia), Some(text)) = (turn["dia_id"].as_str(), turn["text"].as_str())
                    else {
                        continue;
                    };
                    let speaker = turn["speaker"].as_str().unwrap_or("");
                    let memory = if date.is_empty() {
                        format!("{speaker}: {text}")
                    } else {
                        format!("[{date}] {speaker}: {text}")
                    };
                    mem.remember(EgressTier::Open, &memory);
                }
            }
        }

        let mut asked = 0usize;
        for qa in sample_conv["qa"].as_array().into_iter().flatten() {
            if asked >= per_conv {
                break;
            }
            let Some(question) = qa["question"].as_str() else {
                continue;
            };
            let category = qa["category"].as_u64().unwrap_or(0);
            // Category 5 is adversarial/unanswerable — it grades abstention, not grounded recall.
            if category == 5 {
                continue;
            }
            let gold = match &qa["answer"] {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Null => continue,
                other => other.to_string(),
            };
            if gold.trim().is_empty() {
                continue;
            }

            let hits = mem.recall_weighted(
                question,
                Destination::Local,
                topk,
                1_000_000,
                RetrievalWeights::semantic(),
            );
            // Cap each memory's length so the prompt stays bounded (long turns otherwise blow up
            // CPU generation time without adding answer-relevant signal).
            let context: String = hits
                .iter()
                .take(topk)
                .map(|b| {
                    let t: String = b.text.chars().take(240).collect();
                    format!("- {t}\n")
                })
                .collect();

            let answer_prompt = format!(
                "Using ONLY the memories below, answer the question as briefly as possible — a name, \
                 date, place, number, or short phrase. If the memories do not contain the answer, \
                 reply exactly: NO ANSWER.\n\nMemories:\n{context}\nQuestion: {question}\nAnswer:"
            );
            let pred = generate(&agent, &url, &model, &answer_prompt);

            let judge_prompt = format!(
                "Grade whether the predicted answer matches the reference answer for the question. \
                 They match if they state the same fact, even if worded differently, abbreviated, or \
                 with extra detail (e.g. \"7 May 2023\" matches \"May 7th, 2023\"; \"2022\" matches \
                 \"in 2022\"; \"NYC\" matches \"New York City\"). Reply with exactly one word: \
                 CORRECT or INCORRECT.\n\nQuestion: {question}\nReference answer: {gold}\nPredicted \
                 answer: {pred}\nGrade:"
            );
            let ok = verdict_is_correct(&generate(&agent, &url, &model, &judge_prompt));

            correct += ok as usize;
            total += 1;
            asked += 1;
            let e = by_cat.entry(category).or_insert((0, 0));
            e.0 += ok as usize;
            e.1 += 1;

            let q: String = question.chars().take(52).collect();
            let p: String = pred.chars().take(52).collect();
            eprintln!(
                "  [{}] cat{category} {} q=\"{q}\" gold=\"{gold}\" pred=\"{p}\"",
                if ok { "OK  " } else { "MISS" },
                if ok { "✓" } else { "✗" }
            );
        }
    }

    if total == 0 {
        println!("no answerable questions sampled — check LOCOMO_PATH / QA_CONVS");
        return;
    }
    println!("\n  category   accuracy   n");
    for (cat, (c, t)) in &by_cat {
        println!("  cat {cat}      {:.3}      {t}", *c as f64 / *t as f64);
    }
    println!(
        "\n  OVERALL QA accuracy: {:.3}  ({correct}/{total} correct, LLM-judged, semantic retrieval)",
        correct as f64 / total as f64
    );
}

#[cfg(not(all(feature = "secure", feature = "local-embed", feature = "http-embed")))]
fn main() {
    println!(
        "run with `--features secure,local-embed,http-embed`, a local Ollama serving $QA_MODEL, and \
         LOCOMO_PATH set (see the bench header)."
    );
}
