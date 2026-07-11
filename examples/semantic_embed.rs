//! Proof that the bundled semantic embedder actually captures *meaning* — not just token
//! overlap like the default `HashEmbedder`. Run with:
//!
//! ```bash
//! cargo run --example semantic_embed --features local-embed
//! ```
//!
//! On first run it downloads all-MiniLM-L6-v2 from Hugging Face (~90 MB, cached thereafter).
//! It embeds three sentences and prints their cosine similarities: a paraphrase must score
//! higher than an unrelated sentence that happens to share stop-words. A hashed bag-of-words
//! embedder cannot do this (no shared content tokens → ~0 similarity both ways).

use mnema::model_embed::MiniLmEmbedder;
use mnema::vector::Embedder;

/// Cosine of two L2-normalised vectors is just their dot product.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("loading all-MiniLM-L6-v2 (first run downloads it from Hugging Face)…");
    let embedder = MiniLmEmbedder::load()?;
    println!("loaded — {} dims\n", embedder.dims());

    let anchor = "a small cat is sleeping on the sofa";
    let paraphrase = "a kitten naps on the couch"; // same meaning, almost no shared words
    let unrelated = "quarterly earnings sent the stock market lower"; // shares "the", "on"? no

    let a = embedder.embed(anchor);
    let p = embedder.embed(paraphrase);
    let u = embedder.embed(unrelated);

    let sim_para = cosine(&a, &p);
    let sim_unrel = cosine(&a, &u);

    println!("anchor:      {anchor:?}");
    println!("paraphrase:  {paraphrase:?}   cos = {sim_para:.3}");
    println!("unrelated:   {unrelated:?}   cos = {sim_unrel:.3}\n");

    assert!(
        sim_para > sim_unrel,
        "semantic embedder failed: paraphrase ({sim_para:.3}) should beat unrelated ({sim_unrel:.3})"
    );
    println!(
        "✅ semantic recall works: the paraphrase ({sim_para:.3}) outranks the unrelated sentence ({sim_unrel:.3})."
    );
    Ok(())
}
