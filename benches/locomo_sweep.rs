//! **Fusion-weight sweep** on LoCoMo retrieval — find the `RetrievalWeights` that maximize
//! Recall@k, to check (and tune) the `RetrievalWeights::semantic()` default (dense 4.0). Motivated
//! by the QA-accuracy eval (`benches/locomo_qa.rs`): over-weighting the dense retriever may drown
//! out the keyword retriever on exact-match questions (dates, names), hurting recall.
//!
//! Efficient by construction: fusion weights don't change embeddings, so each conversation is
//! ingested **once** through a **memoizing embedder** — every turn and query is embedded a single
//! time, then re-ranked under every candidate weight config for free. The whole multi-config sweep
//! costs one embedding pass (turns + unique queries) per conversation, so it runs in-process over a
//! bounded set of conversations without candle's long-run native-memory buildup.
//!
//! ```bash
//! # quick: sweep the first few conversations in one process
//! LOCOMO_PATH=$PWD/locomo10.json cargo bench --bench locomo_sweep --features secure,local-embed
//! # full 10 conversations (two safe halves, aggregated):
//! LOCOMO_PATH=$PWD/locomo10.json bash scripts/locomo_sweep.sh
//! ```
//! Config: `$SWEEP_START` (first conversation, default 0) and `$SWEEP_CONVS` (count, default 5) —
//! a window, so the full dataset can be run in two halves the driver aggregates.
//!
//! Result (full 10, 1981 questions): balanced fusion (d1/r1/k1) beats the dense-boosted
//! `semantic()` default (d4/r1/k1) — R@5 0.425 vs 0.401, R@10 0.543 vs 0.467 — with no paraphrase
//! regression, which is why the server now recalls with balanced weights.

#[cfg(all(feature = "secure", feature = "local-embed"))]
mod sweep {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use mnema::facade::Mnema;
    use mnema::retrieval::RetrievalWeights;
    use mnema::vector::Embedder;
    use mnema::{Destination, EgressTier};

    /// Wraps an embedder and memoizes `embed` by text, so re-ranking the same store under many weight
    /// configs never re-embeds a turn or query already seen. The expensive model forward pass runs
    /// exactly once per distinct string; the cache is dropped with the store between conversations.
    pub struct Memoizing<E> {
        inner: E,
        cache: RefCell<HashMap<String, Vec<f32>>>,
    }
    impl<E: Embedder> Memoizing<E> {
        fn new(inner: E) -> Self {
            Self {
                inner,
                cache: RefCell::new(HashMap::new()),
            }
        }
    }
    impl<E: Embedder> Embedder for Memoizing<E> {
        fn dims(&self) -> usize {
            self.inner.dims()
        }
        fn embed(&self, text: &str) -> Vec<f32> {
            if let Some(v) = self.cache.borrow().get(text) {
                return v.clone();
            }
            let v = self.inner.embed(text);
            self.cache.borrow_mut().insert(text.to_string(), v.clone());
            v
        }
    }

    /// Pull the `D<session>:<turn>` dia_ids out of LoCoMo's `evidence` (a stringified list) by
    /// scanning for the token shape — same as `benches/locomo.rs`.
    fn evidence_ids(raw: &str) -> Vec<String> {
        let cleaned: String = raw
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == ':' {
                    c
                } else {
                    ' '
                }
            })
            .collect();
        cleaned
            .split_whitespace()
            .filter(|t| {
                let b = t.as_bytes();
                b.first() == Some(&b'D')
                    && t.contains(':')
                    && t[1..].chars().next().is_some_and(|c| c.is_ascii_digit())
            })
            .map(str::to_string)
            .collect()
    }

    /// The candidate weight configs to compare. `semantic()` (dense 4.0) is the current default;
    /// the rest probe whether easing the dense dominance and/or lifting keyword helps recall.
    fn configs() -> Vec<(&'static str, RetrievalWeights)> {
        let w = |dense: f32, recency: f32, keyword: f32| RetrievalWeights {
            dense,
            recency,
            keyword,
        };
        vec![
            ("balanced  d1/r1/k1", w(1.0, 1.0, 1.0)),
            ("semantic  d4/r1/k1", w(4.0, 1.0, 1.0)),
            ("dense2    d2/r1/k1", w(2.0, 1.0, 1.0)),
            ("dense3    d3/r1/k1", w(3.0, 1.0, 1.0)),
            ("dense6    d6/r1/k1", w(6.0, 1.0, 1.0)),
            ("d4+kw2    d4/r1/k2", w(4.0, 1.0, 2.0)),
            ("d3+kw2    d3/r1/k2", w(3.0, 1.0, 2.0)),
            ("d2+kw2    d2/r1/k2", w(2.0, 1.0, 2.0)),
            ("d3+kw2-r0 d3/r0/k2", w(3.0, 0.0, 2.0)),
        ]
    }

    /// The in-repo paraphrase fixture (a copy of `benches/recall.rs`'s pairs): the query means the
    /// same as the memory but shares almost no words, so it is a **pure-semantic** workload — the
    /// opposite of LoCoMo's exact-match-heavy questions. A weight config good for LoCoMo must not
    /// wreck this, or it has just overfit to lexical matching. Kept short; it is a guardrail.
    const PARAPHRASE: &[(&str, &str)] = &[
        (
            "I moved my code editor from VS Code to Neovim last month",
            "which text editor does the user work in these days",
        ),
        (
            "My daughter Mia has a severe peanut allergy",
            "what food must we keep away from the user's kid",
        ),
        (
            "We ship the backend to production every Friday evening",
            "when do releases go live",
        ),
        (
            "I'm training for a marathon in the autumn",
            "what long-distance running event is the user preparing for",
        ),
        (
            "The office coffee machine is broken again",
            "is the espresso maker at work functioning",
        ),
        (
            "I prefer my meetings scheduled in the morning",
            "what time of day suits the user for calls",
        ),
        (
            "Our database runs on PostgreSQL 16",
            "which relational store backs our system",
        ),
        (
            "I adopted a rescue greyhound named Comet",
            "does the user own a pet dog",
        ),
        (
            "The client wants the invoice paid in euros, not dollars",
            "what currency should we bill the customer in",
        ),
        (
            "I usually cycle to work when it isn't raining",
            "how does the user commute on dry days",
        ),
        (
            "My flight to Tokyo departs at 6am on Tuesday",
            "when does the user leave for Japan",
        ),
        (
            "We decided to drop support for Internet Explorer",
            "which legacy browser are we no longer maintaining",
        ),
        (
            "I'm lactose intolerant so I take my coffee black",
            "why does the user avoid milk in drinks",
        ),
        (
            "The staging server lives in the Frankfurt region",
            "where is our pre-production environment hosted",
        ),
        (
            "I switched from Android to an iPhone this year",
            "what kind of smartphone does the user carry now",
        ),
        (
            "Our team stand-up happens at 9:30 every weekday",
            "when is the daily sync",
        ),
        (
            "I keep my savings in an index fund, not individual stocks",
            "how does the user invest their money",
        ),
        (
            "The wedding is booked for the second Saturday in June",
            "when is the user getting married",
        ),
        (
            "I write my notes in Markdown and sync them with git",
            "how does the user keep track of their notes",
        ),
        (
            "We use Stripe to handle customer payments",
            "which service processes our transactions",
        ),
    ];

    /// R@5 over the paraphrase fixture for each config, so a LoCoMo winner can be checked against a
    /// pure-semantic workload in the same run.
    fn paraphrase_r5<E: Embedder>(
        mem: &Mnema<E>,
        ids: &[u64],
        cfgs: &[(&str, RetrievalWeights)],
    ) -> Vec<f64> {
        let mut hits = vec![0usize; cfgs.len()];
        for (i, (_, query)) in PARAPHRASE.iter().enumerate() {
            for (c, (_, weights)) in cfgs.iter().enumerate() {
                let bundle =
                    mem.recall_weighted(query, Destination::Local, 10, 1_000_000, *weights);
                if bundle.iter().take(5).any(|b| b.id == ids[i]) {
                    hits[c] += 1;
                }
            }
        }
        hits.into_iter()
            .map(|h| h as f64 / PARAPHRASE.len() as f64)
            .collect()
    }

    pub fn run() {
        use mnema::model_embed::MiniLmEmbedder;

        let path = std::env::var("LOCOMO_PATH")
            .expect("set LOCOMO_PATH to a downloaded data/locomo10.json (see the bench header)");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read LOCOMO_PATH"))
                .expect("parse LoCoMo json");
        let all = data.as_array().expect("dataset is an array");
        // `SWEEP_START`..`SWEEP_START+SWEEP_CONVS` — a window so the full dataset can be run in two
        // safe halves (candle can't hold all 10 conversations' forward passes in one process). The
        // machine-readable `SUM` lines let a driver aggregate the halves exactly.
        let start: usize = std::env::var("SWEEP_START")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
            .min(all.len());
        let n_convs: usize = std::env::var("SWEEP_CONVS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5)
            .min(all.len() - start);

        let cfgs = configs();
        // per-config running sums: (r5_sum, r10_sum)
        let mut sums = vec![(0.0_f64, 0.0_f64); cfgs.len()];
        let mut questions = 0usize;

        eprintln!(
            "sweeping {} configs over conversations {start}..{}…",
            cfgs.len(),
            start + n_convs
        );
        for (ci, sample) in all.iter().skip(start).take(n_convs).enumerate() {
            let conv = &sample["conversation"];
            // One memoizing model per conversation: every turn/query embeds once, reused across all
            // configs. Fresh per conversation so the cache (and candle's buffers) don't grow forever.
            let mut mem = Mnema::new(Memoizing::new(
                MiniLmEmbedder::load().expect("load all-MiniLM-L6-v2"),
            ));
            let mut id_of: HashMap<String, u64> = HashMap::new();
            if let Some(obj) = conv.as_object() {
                for (key, val) in obj {
                    if !(key.starts_with("session_") && val.is_array()) {
                        continue;
                    }
                    for turn in val.as_array().unwrap() {
                        let (Some(dia), Some(text)) =
                            (turn["dia_id"].as_str(), turn["text"].as_str())
                        else {
                            continue;
                        };
                        let speaker = turn["speaker"].as_str().unwrap_or("");
                        let id = mem.remember(EgressTier::Open, &format!("{speaker}: {text}"));
                        id_of.insert(dia.to_string(), id);
                    }
                }
            }

            for qa in sample["qa"].as_array().into_iter().flatten() {
                let Some(question) = qa["question"].as_str() else {
                    continue;
                };
                let gold_raw = match &qa["evidence"] {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let gold: Vec<u64> = evidence_ids(&gold_raw)
                    .iter()
                    .filter_map(|d| id_of.get(d).copied())
                    .collect();
                if gold.is_empty() {
                    continue;
                }
                let denom = gold.len() as f64;
                questions += 1;

                for (i, (_, weights)) in cfgs.iter().enumerate() {
                    let hits =
                        mem.recall_weighted(question, Destination::Local, 20, 1_000_000, *weights);
                    let ids: Vec<u64> = hits.iter().take(10).map(|b| b.id).collect();
                    let in5 = gold
                        .iter()
                        .filter(|g| ids.iter().take(5).any(|i| i == *g))
                        .count();
                    let in10 = gold.iter().filter(|g| ids.contains(g)).count();
                    sums[i].0 += in5 as f64 / denom;
                    sums[i].1 += in10 as f64 / denom;
                }
            }
            eprintln!("  conversation {ci} done (running n={questions})");
        }

        // Machine-readable per-config sums (r5_sum r10_sum) + question count, so two half-runs can be
        // aggregated into a full-dataset result without re-embedding.
        for (i, (r5s, r10s)) in sums.iter().enumerate() {
            println!("SUM {i} {r5s} {r10s}");
        }
        println!("N {questions}");

        // Rank configs by R@5, print the table with the winner marked.
        let mut order: Vec<usize> = (0..cfgs.len()).collect();
        order.sort_by(|&a, &b| sums[b].0.partial_cmp(&sums[a].0).unwrap());
        let best = order[0];

        println!(
            "\nLoCoMo fusion-weight sweep — mean Recall@k over {questions} answerable questions \
             ({n_convs} conversations)\n"
        );
        // Cross-check every config on the pure-semantic paraphrase fixture (guards against a config
        // that only wins by leaning on LoCoMo's exact-token overlap).
        eprintln!("cross-checking configs on the paraphrase fixture…");
        let mut pmem = Mnema::new(Memoizing::new(
            MiniLmEmbedder::load().expect("load all-MiniLM-L6-v2"),
        ));
        let pids: Vec<u64> = PARAPHRASE
            .iter()
            .map(|(m, _)| pmem.remember(EgressTier::Open, m))
            .collect();
        let para = paraphrase_r5(&pmem, &pids, &cfgs);

        println!("  config                LoCoMo-R@5  LoCoMo-R@10  paraphrase-R@5");
        for &i in &order {
            let (r5, r10) = (sums[i].0 / questions as f64, sums[i].1 / questions as f64);
            let mark = if i == best {
                "  <- best LoCoMo R@5"
            } else {
                ""
            };
            println!(
                "  {}   {r5:.3}       {r10:.3}        {:.3}{mark}",
                cfgs[i].0, para[i]
            );
        }
        let sem = cfgs
            .iter()
            .position(|(n, _)| n.starts_with("semantic"))
            .unwrap();
        println!(
            "\n  best vs semantic() default:  LoCoMo R@5 {:+.1} pts,  paraphrase R@5 {:+.1} pts",
            (sums[best].0 - sums[sem].0) / questions as f64 * 100.0,
            (para[best] - para[sem]) * 100.0
        );
    }
}

#[cfg(all(feature = "secure", feature = "local-embed"))]
fn main() {
    sweep::run();
}

#[cfg(not(all(feature = "secure", feature = "local-embed")))]
fn main() {
    println!(
        "run with `--features secure,local-embed` and LOCOMO_PATH set (see the bench header)."
    );
}
