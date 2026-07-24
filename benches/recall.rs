//! Recall-quality benchmark: does mnema actually *retrieve the right memory* — and does the
//! semantic path (real embedder + dense-weighted fusion) beat the lexical default?
//!
//! Run with the semantic model (the facade needs `secure`, the model needs `local-embed`):
//! ```bash
//! cargo bench --bench recall --features secure,local-embed
//! ```
//!
//! It stores a small, diverse corpus of memories, then for each memory issues a **paraphrase
//! query** that shares almost no words with it (so lexical overlap can't win) but the same
//! meaning. It reports R@1 / R@3 / R@5 — the fraction of queries whose target memory appears in
//! the top-k — for two configurations over the identical corpus and queries:
//!
//!   * **lexical**  — HashEmbedder + balanced fusion (the default)
//!   * **semantic** — all-MiniLM-L6-v2 + dense-weighted fusion (RetrievalWeights::semantic)
//!
//! The gap between the two columns is the value the gap-#1 work (a real embedder + weighted
//! fusion) delivers on the memory layer's actual job. The fixture is in-repo and reviewable —
//! this is mnema's own harness (an honest retrieval measure like mnemo's LoCoMo R@k), not a
//! LongMemEval leaderboard score.

#[cfg(feature = "secure")]
use mnema::facade::Mnema;
#[cfg(feature = "secure")]
use mnema::retrieval::RetrievalWeights;
#[cfg(feature = "secure")]
use mnema::vector::Embedder;
#[cfg(feature = "secure")]
use mnema::{Destination, EgressTier};

/// `(memory, paraphrase-query)` — the query means the same as the memory but is worded
/// differently, so a lexical embedder shares little/nothing to match on.
#[cfg(feature = "secure")]
const PAIRS: &[(&str, &str)] = &[
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
    (
        "My back has been sore since I started sitting all day",
        "why is the user experiencing physical discomfort",
    ),
    (
        "The recipe calls for fresh basil, not the dried kind",
        "what herb does the dish need",
    ),
    (
        "I turned off notifications to focus during deep work",
        "what did the user mute to concentrate",
    ),
    (
        "Our API rate limit is 100 requests per minute",
        "how many calls can a client make each minute",
    ),
    (
        "I grew up in a small fishing town on the coast",
        "where did the user spend their childhood",
    ),
    (
        "The presentation needs to be ready before the board meeting",
        "what deadline is the slide deck tied to",
    ),
];

/// Store the corpus, then measure how often each paraphrase query retrieves its own memory in
/// the top-1 / top-3 / top-5. Returns the three recall fractions.
#[cfg(feature = "secure")]
fn evaluate<E: Embedder>(embedder: E, weights: RetrievalWeights) -> (f64, f64, f64) {
    let mut mem = Mnema::new(embedder);
    let ids: Vec<u64> = PAIRS
        .iter()
        .map(|(content, _)| mem.remember(EgressTier::Open, content))
        .collect();

    let (mut r1, mut r3, mut r5) = (0usize, 0usize, 0usize);
    for (i, (_, query)) in PAIRS.iter().enumerate() {
        let gold = ids[i];
        // A budget large enough that the top-k is never truncated for this small corpus.
        let bundle = mem.recall_weighted(query, Destination::Local, 10, 1_000_000, weights);
        if let Some(rank) = bundle.iter().position(|b| b.id == gold) {
            if rank < 1 {
                r1 += 1;
            }
            if rank < 3 {
                r3 += 1;
            }
            if rank < 5 {
                r5 += 1;
            }
        }
    }
    let n = PAIRS.len() as f64;
    (r1 as f64 / n, r3 as f64 / n, r5 as f64 / n)
}

#[cfg(all(feature = "secure", feature = "local-embed"))]
fn main() {
    use mnema::embed::HashEmbedder;
    use mnema::model_embed::MiniLmEmbedder;

    println!(
        "recall over {} paraphrase queries (target memory in top-k):\n",
        PAIRS.len()
    );
    println!("  config     R@1     R@3     R@5");

    let (l1, l3, l5) = evaluate(
        HashEmbedder::new(HashEmbedder::DEFAULT_DIMS),
        RetrievalWeights::default(),
    );
    println!("  lexical    {l1:.3}   {l3:.3}   {l5:.3}");

    let model = MiniLmEmbedder::load().expect("load all-MiniLM-L6-v2");
    let (s1, s3, s5) = evaluate(model, RetrievalWeights::semantic());
    println!("  semantic   {s1:.3}   {s3:.3}   {s5:.3}");

    println!(
        "\nR@5 lift from the semantic path: {:+.1} percentage points",
        (s5 - l5) * 100.0
    );
}

#[cfg(all(feature = "secure", not(feature = "local-embed")))]
fn main() {
    // The lexical baseline alone is not the point; the comparison needs the model.
    let (l1, l3, l5) = evaluate(
        mnema::embed::HashEmbedder::new(mnema::embed::HashEmbedder::DEFAULT_DIMS),
        RetrievalWeights::default(),
    );
    println!("lexical-only R@1/R@3/R@5 = {l1:.3}/{l3:.3}/{l5:.3}");
    println!("re-run with `--features local-embed` to compare against the semantic path.");
}

// The corpus is stored through the `Mnema` facade, which lives behind `secure`.
#[cfg(not(feature = "secure"))]
fn main() {
    println!("run with `--features secure,local-embed` (see the bench header).");
}
