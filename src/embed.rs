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
    /// truth for it. Every process that touches one store family — the `mnema` CLI and
    /// the `mnema-mcp` server both do — must embed at the *same* width: a query vector
    /// of a different length than the stored vectors makes cosine similarity meaningless
    /// and silently corrupts recall. Pinning it here, rather than as a private `const` in
    /// each binary, makes that agreement structural instead of a convention two crates can
    /// drift apart on. See ADR-0023.
    pub const DEFAULT_DIMS: usize = 128;

    /// A new embedder producing `dims`-dimensional vectors. `dims == 0` is a degenerate
    /// embedder that maps every text to the empty vector (see [`embed`](HashEmbedder::embed)) —
    /// it never panics, but it also carries no signal, so use [`DEFAULT_DIMS`](HashEmbedder::DEFAULT_DIMS)
    /// unless you have a reason not to.
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Embedder for HashEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // A zero-width embedder has no bucket to land a token in; `h % 0` would panic, so
        // fall through to the empty vector rather than divide by zero. (The default width
        // is DEFAULT_DIMS; this only guards the degenerate `new(0)`.)
        if self.dims == 0 {
            return Vec::new();
        }
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

/// Mean-pool a sequence of per-token embedding vectors using an attention `mask`: average
/// only the real tokens (`mask == 1`), skipping padding (`mask == 0`). This is the standard
/// sentence-embedding reduction for models like all-MiniLM-L6-v2 — the model emits one vector
/// per token; a sentence embedding is the mean of the non-padding ones. Exposed (with
/// [`l2_normalize`]) so a transformer-backed [`Embedder`](crate::vector::Embedder) can reuse the
/// pooling math this crate has tested rather than re-deriving it. A zero-length input yields an
/// empty vector; an all-padding mask yields the zero vector (no division by zero).
pub fn mean_pool(token_embeddings: &[Vec<f32>], mask: &[u32]) -> Vec<f32> {
    let dims = token_embeddings.first().map_or(0, Vec::len);
    let mut sum = vec![0.0f32; dims];
    let mut count = 0u32;
    for (tok, &m) in token_embeddings.iter().zip(mask) {
        if m == 0 {
            continue;
        }
        count += 1;
        for (s, &x) in sum.iter_mut().zip(tok) {
            *s += x;
        }
    }
    if count > 0 {
        for s in &mut sum {
            *s /= count as f32;
        }
    }
    sum
}

/// L2-normalize `v` so its Euclidean length is 1, making cosine similarity a plain dot
/// product. A zero vector is returned unchanged (its direction is undefined — never divide by
/// zero). Pairs with [`mean_pool`] for transformer sentence embeddings.
pub fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_only_the_unmasked_tokens() {
        let tokens = vec![vec![1.0, 1.0], vec![3.0, 3.0], vec![9.0, 9.0]];
        // mask drops the third token → mean of [1,1] and [3,3] = [2,2] (count is 2, not 3).
        assert_eq!(mean_pool(&tokens, &[1, 1, 0]), vec![2.0, 2.0]);
        // all-padding → the zero vector, never a divide-by-zero.
        assert_eq!(mean_pool(&tokens, &[0, 0, 0]), vec![0.0, 0.0]);
        // no tokens at all → empty.
        assert!(mean_pool(&[], &[]).is_empty());
    }

    #[test]
    fn l2_normalize_scales_to_unit_length_and_guards_zero() {
        // [3,4] has norm 5 → [0.6, 0.8].
        let n = l2_normalize(vec![3.0, 4.0]);
        assert!(
            (n[0] - 0.6).abs() < 1e-6 && (n[1] - 0.8).abs() < 1e-6,
            "{n:?}"
        );
        assert!(
            (n.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6,
            "unit length"
        );
        // a zero vector is returned unchanged.
        assert_eq!(l2_normalize(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }

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
    fn a_zero_width_embedder_yields_the_empty_vector_without_panicking() {
        // `new(0)` used to panic on the first token (`h % 0`); it must now be a total,
        // signal-free embedder instead — no divide-by-zero on any input.
        let e = HashEmbedder::new(0);
        assert_eq!(e.dims(), 0);
        assert!(e.embed("anything at all").is_empty());
        assert!(e.embed("").is_empty());
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
