//! Persistence cost: whole-store `seal()` latency as the corpus grows, with the derived key
//! already cached (so this is the encode + XChaCha encrypt, NOT the one-time Argon2id KDF).
//!
//! engram persists the **whole store** per write rather than appending to a log — a deliberate
//! choice so that `forget` is a *true immediate physical delete* (the rewritten blob simply does
//! not contain the forgotten bytes). See [ADR-0024]. This benchmark is the evidence that the
//! whole-store write stays cheap at this layer's scale once the KDF is cached; if it stopped
//! being cheap at a realistic N, that would be the trigger to revisit the decision.
//!
//! Run: `cargo bench --bench persist --features secure`
//!
//! [ADR-0024]: ../../docs/adr/0024-whole-store-seal-persistence.md

#[cfg(feature = "secure")]
fn main() {
    use engram::EgressTier;
    use engram::embed::HashEmbedder;
    use engram::facade::Engram;
    use std::time::Instant;

    let key = b"persist-bench-key";
    println!("whole-store seal() latency vs N (derived key cached — no per-write Argon2id):\n");
    println!("        N    seal_ms");
    for &n in &[1_000u64, 10_000, 50_000, 100_000] {
        let mut e = Engram::new(HashEmbedder::new(8));
        for _ in 0..n {
            e.remember(EgressTier::Open, "a memory about various everyday topics and things");
        }
        let _ = e.seal(key).unwrap(); // prime the key cache (pays the one-time KDF once)
        let mut samples = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let blob = e.seal(key).unwrap();
            samples.push(t.elapsed().as_micros() as u64);
            std::hint::black_box(blob);
        }
        samples.sort_unstable();
        println!("{n:>9}   {:>7.2}", samples[2] as f64 / 1000.0);
    }
}

#[cfg(not(feature = "secure"))]
fn main() {
    println!("run with `--features secure` — seal() lives behind the secure feature.");
}
