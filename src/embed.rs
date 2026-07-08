//! A built-in, dependency-free default [`Embedder`]: a hashed bag-of-words. Each token
//! bumps one dimension (FNV-1a → bucket), so texts that share words land near each
//! other. It is *stable* — like the in-repo `FastHasher` (ADR-0012, "own your hasher"),
//! its exact mixing is pinned by a golden-witness test, so the vectors it produces can
//! never silently drift and equivalent mutants stay killable.
//!
//! This is not a semantic model — it is a zero-dependency default so the layer is
//! usable out of the box; bring a real local model behind the [`Embedder`] seam
//! (ADR-0020) when you need true semantics.

use crate::vector::Embedder;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A hashed bag-of-words embedder over a fixed number of dimensions.
#[derive(Clone, Copy, Debug)]
pub struct HashEmbedder {
    dims: usize,
}

impl HashEmbedder {
    /// The canonical width for the built-in default embedder, and the single source of
    /// truth for it. Every process that touches one store family — the `engram` CLI and
    /// the `engram-mcp` server both do — must embed at the *same* width: a query vector
    /// of a different length than the stored vectors makes cosine similarity meaningless
    /// and silently corrupts recall. Pinning it here, rather than as a private `const` in
    /// each binary, makes that agreement structural instead of a convention two crates can
    /// drift apart on. See ADR-0023.
    pub const DEFAULT_DIMS: usize = 128;

    /// A new embedder producing `dims`-dimensional vectors.
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Embedder for HashEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for token in text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
        {
            let mut h = FNV_OFFSET;
            for b in token.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(FNV_PRIME);
            }
            v[(h % self.dims as u64) as usize] += 1.0;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_token_adds_one_to_exactly_one_dimension() {
        let e = HashEmbedder::new(16);
        // Three distinct tokens → the vector sums to 3 (pins the `+= 1.0` accumulation).
        let v = e.embed("alpha beta gamma");
        let sum: f32 = v.iter().sum();
        assert_eq!(sum, 3.0);
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn a_repeated_token_stacks_in_its_own_bucket() {
        let e = HashEmbedder::new(16);
        let v = e.embed("cat cat cat");
        // The same token hashes to the same dimension every time → one dim holds 3.0.
        assert!(v.contains(&3.0));
        assert_eq!(v.iter().sum::<f32>(), 3.0);
    }

    #[test]
    fn embedding_is_case_insensitive_and_punctuation_split() {
        let e = HashEmbedder::new(16);
        assert_eq!(e.embed("Cat, DOG!"), e.embed("cat dog"));
    }

    #[test]
    fn golden_witness_pins_the_exact_mixing() {
        // Like ADR-0012's FastHasher witness: the exact bucket a token lands in pins the
        // FNV mixing (`^=`, `wrapping_mul`, `% dims`). A mutation to any of those moves a
        // token to a different dimension and breaks this — no equivalent-mutant escape.
        let e = HashEmbedder::new(8);
        assert_eq!(e.embed("cat"), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(e.embed("dog"), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
