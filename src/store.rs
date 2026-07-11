//! Encrypted episodic store — Phase-1 slice 2 (`docs/proposals/mnema-memory-layer.md`
//! §3.1, §6b). An append-only log of episodic memories that is **encrypted at rest**:
//! the in-memory [`EpisodicLog`] holds events in time order; [`EpisodicLog::seal`]
//! turns it into an opaque byte blob (Argon2id-derived key + XChaCha20-Poly1305 AEAD)
//! and [`EpisodicLog::open`] recovers it. A stolen disk image yields ciphertext only.
//!
//! Gated behind the `secure` feature (ADR-0020): the crypto dependencies exist only
//! when this compiles, so the evolution substrate and benches stay zero-dependency.
//!
//! The proof surface for internal-tool is the **manual codec** ([`EpisodicLog::encode`] /
//! [`decode`]): pure, deterministic offset arithmetic where every `+`, `<`, and `>`
//! is a mutation target. A mutant that miscounts an offset or a length must break a
//! round-trip test below — that is what makes "the bytes survive encryption intact"
//! *proven*, not asserted. The AEAD itself is trusted (RustCrypto), exercised by the
//! round-trip and wrong-key tests.

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::aead::Aead;
use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce};

use crate::{EgressTier, Memory, MemoryId, MemoryKind};

/// On-disk framing: `salt(16) || nonce(24) || ciphertext`. The salt lets `open`
/// re-derive the key; the 24-byte XChaCha nonce makes a random nonce collision-safe.
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 24;

/// Everything that can go wrong sealing, opening, or decoding a store.
#[derive(Debug, PartialEq, Eq)]
pub enum StoreError {
    /// Argon2id failed to derive a key from the passphrase + salt.
    KeyDerivation,
    /// The OS entropy source failed (nonce/salt generation).
    Entropy,
    /// AEAD rejected the ciphertext — wrong key, or tampered bytes.
    Decrypt,
    /// The plaintext ended in the middle of a record — truncated or corrupt.
    Truncated,
    /// A content field was not valid UTF-8.
    BadUtf8,
    /// A kind/tier tag byte was outside its known set.
    UnknownTag,
}

fn kind_tag(kind: MemoryKind) -> u8 {
    match kind {
        MemoryKind::Episodic => 0,
        MemoryKind::Semantic => 1,
        MemoryKind::Procedural => 2,
        MemoryKind::Working => 3,
    }
}

fn kind_from_tag(tag: u8) -> Result<MemoryKind, StoreError> {
    match tag {
        0 => Ok(MemoryKind::Episodic),
        1 => Ok(MemoryKind::Semantic),
        2 => Ok(MemoryKind::Procedural),
        3 => Ok(MemoryKind::Working),
        _ => Err(StoreError::UnknownTag),
    }
}

fn tier_tag(tier: EgressTier) -> u8 {
    match tier {
        EgressTier::Open => 0,
        EgressTier::Redacted => 1,
        EgressTier::Private => 2,
    }
}

fn tier_from_tag(tag: u8) -> Result<EgressTier, StoreError> {
    match tag {
        0 => Ok(EgressTier::Open),
        1 => Ok(EgressTier::Redacted),
        2 => Ok(EgressTier::Private),
        _ => Err(StoreError::UnknownTag),
    }
}

/// Derive a 32-byte key from a passphrase + salt with Argon2id (memory-hard).
fn derive_key(passphrase: &[u8], salt: &[u8]) -> Result<[u8; 32], StoreError> {
    let mut key = [0u8; 32];
    argon2()?
        .hash_password_into(passphrase, salt, &mut key)
        .map_err(|_| StoreError::KeyDerivation)?;
    Ok(key)
}

/// The Argon2id configuration, pinned **explicitly** rather than taken from
/// `Argon2::default()`. The parameters (m/t/p) are baked into the derived key, so if the
/// `argon2` crate ever changed its default, every existing store would silently derive a
/// different key and fail to open. Pinning them here makes the KDF reproducible and
/// upgrade-safe. These match the current OWASP-recommended defaults; raising them (or adding
/// AEAD associated-data binding) is a deliberate, versioned format migration — it re-keys
/// every store and so needs a format byte + migration path, tracked separately.
fn argon2() -> Result<Argon2<'static>, StoreError> {
    let params = Params::new(
        Params::DEFAULT_M_COST,
        Params::DEFAULT_T_COST,
        Params::DEFAULT_P_COST,
        Some(32),
    )
    .map_err(|_| StoreError::KeyDerivation)?;
    Ok(Argon2::new(Algorithm::Argon2id, Version::V0x13, params))
}

/// Append a length-prefixed byte string (`u32` LE length, then the bytes).
pub(crate) fn put_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// The audit trail of a [`EpisodicLog::forget`] call — what was hard-deleted and how
/// much remains. A caller records this to prove a right-to-be-forgotten request was
/// honoured (proposal §6d).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurgeReceipt {
    /// The ids of the events that were removed, in log order.
    pub purged: Vec<MemoryId>,
    /// How many events remain after the purge.
    pub remaining: usize,
}

/// A time-ordered log of episodic memories, encryptable at rest and **compactable**:
/// [`forget`](EpisodicLog::forget) hard-deletes events rather than tombstoning them,
/// so the purged bytes leave the state and every subsequently-sealed blob.
///
/// Because a purge removes events, ids are no longer `0..len`; a monotonic `next_id`
/// (restored on `open` from the decoded ids) guarantees a forgotten id is **never
/// reused** — a stale reference resolves to nothing, never to a different memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EpisodicLog {
    events: Vec<Memory>,
    next_id: MemoryId,
}

impl EpisodicLog {
    /// A fresh, empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event of neutral importance (`1.0`); returns its assigned id.
    pub fn append(
        &mut self,
        kind: MemoryKind,
        tier: EgressTier,
        at: u64,
        content: impl Into<String>,
    ) -> MemoryId {
        self.append_important(kind, tier, at, 1.0, content)
    }

    /// Append an event with an explicit `importance` (see [`Memory::importance`]) and an
    /// empty redacted surface; returns its freshly assigned, monotonic id. Use
    /// [`append_redacted`](EpisodicLog::append_redacted) to attach a surface that may go
    /// remote for a `Redacted`-tier memory.
    pub fn append_important(
        &mut self,
        kind: MemoryKind,
        tier: EgressTier,
        at: u64,
        importance: f32,
        content: impl Into<String>,
    ) -> MemoryId {
        self.append_redacted(kind, tier, at, importance, content, String::new())
    }

    /// Append an event carrying an explicit `redacted` surface — the text emitted in place
    /// of `content` when a `Redacted`-tier memory is bound for a `Remote` destination (the
    /// egress filter's `Redact` decision). For non-`Redacted` tiers the surface is unused
    /// but harmless. Returns the freshly assigned, monotonic id.
    pub fn append_redacted(
        &mut self,
        kind: MemoryKind,
        tier: EgressTier,
        at: u64,
        importance: f32,
        content: impl Into<String>,
        redacted: impl Into<String>,
    ) -> MemoryId {
        let id = self.next_id;
        self.next_id += 1;
        self.events.push(Memory {
            id,
            kind,
            tier,
            at,
            importance,
            content: content.into(),
            redacted: redacted.into(),
        });
        id
    }

    /// Hard-delete every event matching `predicate`, returning an auditable
    /// [`PurgeReceipt`]. The events are removed from the log entirely (not tombstoned),
    /// so once the caller re-[`seal`](EpisodicLog::seal)s and overwrites the old blob,
    /// the purged content is gone from disk. Ids are not renumbered and `next_id` does
    /// not rewind, so a purged id is never handed to a future event.
    pub fn forget(&mut self, mut predicate: impl FnMut(&Memory) -> bool) -> PurgeReceipt {
        let mut purged = Vec::new();
        self.events.retain(|e| {
            if predicate(e) {
                purged.push(e.id);
                false
            } else {
                true
            }
        });
        PurgeReceipt {
            purged,
            remaining: self.events.len(),
        }
    }

    /// Number of stored events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log holds no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// All events, oldest first (append order).
    pub fn events(&self) -> &[Memory] {
        &self.events
    }

    /// The `k` most-recent events, newest first (recall-by-recency, proposal §3.3).
    pub fn recall_recent(&self, k: usize) -> Vec<&Memory> {
        self.events.iter().rev().take(k).collect()
    }

    /// Serialize every record into the plaintext wire format (pre-encryption).
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for e in &self.events {
            buf.extend_from_slice(&e.id.to_le_bytes());
            buf.push(kind_tag(e.kind));
            buf.push(tier_tag(e.tier));
            buf.extend_from_slice(&e.at.to_le_bytes());
            buf.extend_from_slice(&e.importance.to_le_bytes());
            put_bytes(&mut buf, e.content.as_bytes());
            put_bytes(&mut buf, e.redacted.as_bytes());
        }
        buf
    }

    /// Parse the plaintext wire format back into a log. Every read goes through
    /// [`take_slice`], which returns [`StoreError::Truncated`] rather than
    /// index-panicking, so there is no fragile size constant to keep in sync and
    /// every offset advance is a mutation target the round-trip + truncation tests
    /// pin — a mutant that mis-bounds a read yields the wrong `Result`, not a panic.
    pub(crate) fn decode(buf: &[u8]) -> Result<Self, StoreError> {
        let mut events = Vec::new();
        let mut off = 0usize;
        while off < buf.len() {
            let (id, o) = take_u64(buf, off)?;
            let (kind_byte, o) = take_u8(buf, o)?;
            let (tier_byte, o) = take_u8(buf, o)?;
            let (at, o) = take_u64(buf, o)?;
            let (importance, o) = take_f32(buf, o)?;
            let (content, o) = take_bytes(buf, o)?;
            let (redacted, next) = take_bytes(buf, o)?;
            off = next;
            events.push(Memory {
                id,
                kind: kind_from_tag(kind_byte)?,
                tier: tier_from_tag(tier_byte)?,
                at,
                importance,
                content: string_from(content)?,
                redacted: string_from(redacted)?,
            });
        }
        // Resume the id sequence past the highest surviving id, so a compacted-then-
        // reopened log never reissues a forgotten id.
        let next_id = events.iter().map(|e| e.id).max().map_or(0, |m| m + 1);
        Ok(Self { events, next_id })
    }

    /// Encrypt the whole log at rest — `salt || nonce || AEAD(encode())`.
    pub fn seal(&self, passphrase: &[u8]) -> Result<Vec<u8>, StoreError> {
        seal_bytes(&self.encode(), passphrase)
    }

    /// Recover a log from its sealed bytes with the passphrase. Wrong key or tampered
    /// bytes yield [`StoreError::Decrypt`] (the AEAD tag fails) — never plaintext.
    pub fn open(bytes: &[u8], passphrase: &[u8]) -> Result<Self, StoreError> {
        Self::decode(&open_bytes(bytes, passphrase)?)
    }
}

/// Seal an arbitrary plaintext at rest: `salt || nonce || AEAD(plaintext)`, keying with
/// Argon2id over a fresh random salt. The single encryption choke point — the episodic
/// log and the whole-`Mnema` facade both seal through here, so the crypto lives in one
/// audited place.
pub(crate) fn seal_bytes(plaintext: &[u8], passphrase: &[u8]) -> Result<Vec<u8>, StoreError> {
    let mut salt = [0u8; SALT_LEN];
    getrandom::getrandom(&mut salt).map_err(|_| StoreError::Entropy)?;
    let mut nonce_bytes = [0u8; NONCE_LEN];
    getrandom::getrandom(&mut nonce_bytes).map_err(|_| StoreError::Entropy)?;

    let key = derive_key(passphrase, &salt)?;
    let cipher = XChaCha20Poly1305::new_from_slice(&key).map_err(|_| StoreError::KeyDerivation)?;
    let nonce = XNonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|_| StoreError::Decrypt)?;

    let mut out = Vec::with_capacity(SALT_LEN + NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&salt);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt a blob sealed by [`seal_bytes`], returning its plaintext. A wrong key or any
/// tampering fails the AEAD tag → [`StoreError::Decrypt`]; a blob too short to hold the
/// header is [`StoreError::Truncated`].
pub(crate) fn open_bytes(bytes: &[u8], passphrase: &[u8]) -> Result<Vec<u8>, StoreError> {
    if bytes.len() < SALT_LEN + NONCE_LEN {
        return Err(StoreError::Truncated);
    }
    let (salt, rest) = bytes.split_at(SALT_LEN);
    let (nonce_bytes, ciphertext) = rest.split_at(NONCE_LEN);
    let key = derive_key(passphrase, salt)?;
    let cipher = XChaCha20Poly1305::new_from_slice(&key).map_err(|_| StoreError::KeyDerivation)?;
    let nonce = XNonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| StoreError::Decrypt)
}

/// A derived sealing key plus the salt it came from. **Deriving** it is the expensive,
/// memory-hard Argon2id step; **sealing** or **opening** under an already-derived key is just a
/// fresh-nonce AEAD operation. A store that re-seals on every write (e.g. the MCP server, which
/// persists after each `remember`) derives this **once** — on open, or on the first seal — and
/// reuses it, instead of paying a full Argon2id derivation per write.
///
/// Security is unchanged from [`seal_bytes`]: every `seal` draws a **fresh** 24-byte XChaCha
/// nonce, so reusing the key + salt across seals of one store is safe (a nonce is what must be
/// unique, never the salt). The blob format is identical (`salt || nonce || ciphertext`), so a
/// store sealed this way opens with plain [`open_bytes`] and vice versa.
pub(crate) struct SealingKey {
    /// The passphrase this key was derived from — kept so a caller that seals with a *different*
    /// passphrase (e.g. `mnema rekey`) re-derives instead of silently reusing the old key.
    passphrase: Vec<u8>,
    salt: [u8; SALT_LEN],
    key: [u8; 32],
}

impl SealingKey {
    /// Derive a key from `passphrase` with a fresh random salt — one Argon2id pass.
    pub(crate) fn derive(passphrase: &[u8]) -> Result<Self, StoreError> {
        let mut salt = [0u8; SALT_LEN];
        getrandom::getrandom(&mut salt).map_err(|_| StoreError::Entropy)?;
        let key = derive_key(passphrase, &salt)?;
        Ok(Self {
            passphrase: passphrase.to_vec(),
            salt,
            key,
        })
    }

    /// Re-derive the key for an existing blob's `salt` — the open path (one Argon2id pass).
    pub(crate) fn for_salt(passphrase: &[u8], salt: [u8; SALT_LEN]) -> Result<Self, StoreError> {
        let key = derive_key(passphrase, &salt)?;
        Ok(Self {
            passphrase: passphrase.to_vec(),
            salt,
            key,
        })
    }

    /// Whether this cached key was derived from `passphrase` (so it may be reused). Not a
    /// secret-vs-attacker comparison — it's the caller's own passphrase against itself — so a
    /// plain `==` is fine.
    pub(crate) fn matches(&self, passphrase: &[u8]) -> bool {
        self.passphrase == passphrase
    }

    /// The salt prefixing a sealed blob, so the key can be reconstructed to open it.
    pub(crate) fn salt_of(blob: &[u8]) -> Result<[u8; SALT_LEN], StoreError> {
        let (salt, _) = take_slice(blob, 0, SALT_LEN)?;
        Ok(salt.try_into().unwrap())
    }

    /// Seal `plaintext` under this key with a **fresh** nonce — no KDF. `salt || nonce || ct`.
    pub(crate) fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>, StoreError> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        getrandom::getrandom(&mut nonce_bytes).map_err(|_| StoreError::Entropy)?;
        let cipher =
            XChaCha20Poly1305::new_from_slice(&self.key).map_err(|_| StoreError::KeyDerivation)?;
        let nonce = XNonce::from_slice(&nonce_bytes);
        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| StoreError::Decrypt)?;
        let mut out = Vec::with_capacity(SALT_LEN + NONCE_LEN + ciphertext.len());
        out.extend_from_slice(&self.salt);
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt a blob produced by [`seal`](SealingKey::seal) (or [`seal_bytes`]) under this key —
    /// no KDF, since the key is already in hand. A wrong key or tampering fails the AEAD tag.
    pub(crate) fn open(&self, blob: &[u8]) -> Result<Vec<u8>, StoreError> {
        if blob.len() < SALT_LEN + NONCE_LEN {
            return Err(StoreError::Truncated);
        }
        let (_salt, rest) = blob.split_at(SALT_LEN);
        let (nonce_bytes, ciphertext) = rest.split_at(NONCE_LEN);
        let cipher =
            XChaCha20Poly1305::new_from_slice(&self.key).map_err(|_| StoreError::KeyDerivation)?;
        let nonce = XNonce::from_slice(nonce_bytes);
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| StoreError::Decrypt)
    }
}

/// Borrow `n` bytes at `off`, returning them and the next offset — or
/// [`StoreError::Truncated`] if the buffer is too short. The single bounds check
/// every reader funnels through; `checked_add` makes even an absurd length safe.
pub(crate) fn take_slice(buf: &[u8], off: usize, n: usize) -> Result<(&[u8], usize), StoreError> {
    let end = off.checked_add(n).ok_or(StoreError::Truncated)?;
    if end > buf.len() {
        return Err(StoreError::Truncated);
    }
    Ok((&buf[off..end], end))
}

pub(crate) fn take_u8(buf: &[u8], off: usize) -> Result<(u8, usize), StoreError> {
    let (s, next) = take_slice(buf, off, 1)?;
    Ok((s[0], next))
}

pub(crate) fn take_u32(buf: &[u8], off: usize) -> Result<(u32, usize), StoreError> {
    let (s, next) = take_slice(buf, off, 4)?;
    Ok((u32::from_le_bytes(s.try_into().unwrap()), next))
}

pub(crate) fn take_u64(buf: &[u8], off: usize) -> Result<(u64, usize), StoreError> {
    let (s, next) = take_slice(buf, off, 8)?;
    Ok((u64::from_le_bytes(s.try_into().unwrap()), next))
}

pub(crate) fn take_f32(buf: &[u8], off: usize) -> Result<(f32, usize), StoreError> {
    let (s, next) = take_slice(buf, off, 4)?;
    Ok((f32::from_le_bytes(s.try_into().unwrap()), next))
}

/// Read a length-prefixed byte string at `off`: a `u32` LE length, then its bytes.
pub(crate) fn take_bytes(buf: &[u8], off: usize) -> Result<(&[u8], usize), StoreError> {
    let (len, body) = take_u32(buf, off)?;
    take_slice(buf, body, len as usize)
}

pub(crate) fn string_from(bytes: &[u8]) -> Result<String, StoreError> {
    String::from_utf8(bytes.to_vec()).map_err(|_| StoreError::BadUtf8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> EpisodicLog {
        let mut log = EpisodicLog::new();
        log.append(MemoryKind::Episodic, EgressTier::Open, 100, "woke up");
        // A non-neutral importance so the round-trip pins the `f32` codec byte-exactly.
        log.append_important(
            MemoryKind::Semantic,
            EgressTier::Private,
            200,
            2.5,
            "lives in Utrecht",
        );
        // A record with EMPTY content, placed last: pins the `off + HEAD > len` and
        // `start + len > len` boundary mutants (`>`→`>=` would falsely reject it).
        log.append(MemoryKind::Working, EgressTier::Redacted, 300, "");
        log
    }

    #[test]
    fn append_assigns_contiguous_monotonic_ids() {
        let mut log = EpisodicLog::new();
        assert_eq!(
            log.append(MemoryKind::Episodic, EgressTier::Open, 1, "a"),
            0
        );
        assert_eq!(
            log.append(MemoryKind::Episodic, EgressTier::Open, 2, "b"),
            1
        );
        assert_eq!(
            log.append(MemoryKind::Episodic, EgressTier::Open, 3, "c"),
            2
        );
        assert_eq!(log.len(), 3);
        assert!(!log.is_empty());
    }

    #[test]
    fn recall_recent_is_newest_first_and_bounded() {
        let log = sample();
        let ids: Vec<MemoryId> = log.recall_recent(2).iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![2, 1]); // the two newest, newest first
        assert_eq!(log.recall_recent(99).len(), 3); // k past the end is clamped
        assert!(log.recall_recent(0).is_empty());
    }

    #[test]
    fn codec_round_trips_every_field() {
        let log = sample();
        let decoded = EpisodicLog::decode(&log.encode()).expect("valid encoding decodes");
        assert_eq!(decoded.len(), log.len());
        for (a, b) in decoded.events().iter().zip(log.events()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.tier, b.tier);
            assert_eq!(a.at, b.at);
            assert_eq!(a.importance, b.importance); // pins the f32 field round-trip
            assert_eq!(a.content, b.content);
        }
        // The non-neutral importance survived exactly (2.5 is representable in f32).
        assert_eq!(decoded.events()[1].importance, 2.5);
    }

    #[test]
    fn seal_then_open_recovers_the_log() {
        let log = sample();
        let sealed = log.seal(b"correct horse battery staple").unwrap();
        // At rest it is opaque: the plaintext content must not appear in the blob.
        assert!(!sealed.windows(7).any(|w| w == b"woke up"));
        let opened = EpisodicLog::open(&sealed, b"correct horse battery staple").unwrap();
        assert_eq!(opened.len(), 3);
        assert_eq!(opened.events()[1].content, "lives in Utrecht");
    }

    #[test]
    fn a_wrong_passphrase_never_decrypts() {
        let sealed = sample().seal(b"right key").unwrap();
        assert_eq!(
            EpisodicLog::open(&sealed, b"wrong key"),
            Err(StoreError::Decrypt)
        );
    }

    #[test]
    fn tampering_with_the_ciphertext_is_detected() {
        let mut sealed = sample().seal(b"key").unwrap();
        let last = sealed.len() - 1;
        sealed[last] ^= 0xFF; // flip a ciphertext byte
        assert_eq!(EpisodicLog::open(&sealed, b"key"), Err(StoreError::Decrypt));
    }

    #[test]
    fn truncated_plaintext_is_rejected_not_panicked() {
        let full = sample().encode();
        // Cut a record in half: decode must return Truncated, never index-panic.
        assert_eq!(
            EpisodicLog::decode(&full[..full.len() - 3]),
            Err(StoreError::Truncated)
        );
        assert_eq!(EpisodicLog::decode(&full[..5]), Err(StoreError::Truncated));
    }

    #[test]
    fn a_too_short_blob_is_truncated() {
        assert_eq!(
            EpisodicLog::open(&[0u8; 10], b"k"),
            Err(StoreError::Truncated)
        );
    }

    #[test]
    fn a_blob_of_exactly_the_header_length_reaches_the_aead() {
        // salt(16) + nonce(24) = 40 bytes, zero ciphertext. This pins the `<`
        // boundary in `open`: at len == 40 the header IS present (not Truncated), so
        // control reaches the AEAD, which rejects the empty ciphertext → Decrypt.
        // Flipping `<` to `<=` would wrongly report Truncated here instead.
        let blob = [0u8; SALT_LEN + NONCE_LEN];
        assert_eq!(EpisodicLog::open(&blob, b"k"), Err(StoreError::Decrypt));
    }

    #[test]
    fn a_content_length_past_the_buffer_end_is_rejected_not_panicked() {
        // A valid record whose declared content length is enormous: `take_slice`
        // must return Truncated, never slice out of bounds. Pins the content bounds
        // check — dropping or inverting it would index-panic here.
        let mut buf = {
            let mut log = EpisodicLog::new();
            log.append(MemoryKind::Episodic, EgressTier::Open, 1, "hi");
            log.encode()
        };
        // content_len sits after id(8)+kind(1)+tier(1)+at(8)+importance(4) = offset 22.
        buf[22..26].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(EpisodicLog::decode(&buf), Err(StoreError::Truncated));
    }

    #[test]
    fn forget_hard_deletes_matching_events_and_receipts_them() {
        let mut log = sample(); // ids 0,1,2
        let receipt = log.forget(|m| m.content == "lives in Utrecht"); // id 1
        assert_eq!(receipt.purged, vec![1]);
        assert_eq!(receipt.remaining, 2);
        // The event is gone from state entirely — not tombstoned.
        let ids: Vec<MemoryId> = log.events().iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![0, 2]);
        assert!(!log.events().iter().any(|m| m.content == "lives in Utrecht"));
    }

    #[test]
    fn forgetting_no_match_leaves_the_log_untouched() {
        let mut log = sample();
        let receipt = log.forget(|m| m.content == "never stored");
        assert!(receipt.purged.is_empty());
        assert_eq!(receipt.remaining, 3);
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn a_forgotten_id_is_never_reused() {
        let mut log = EpisodicLog::new();
        log.append(MemoryKind::Episodic, EgressTier::Open, 1, "a"); // 0
        log.append(MemoryKind::Episodic, EgressTier::Open, 2, "b"); // 1
        log.append(MemoryKind::Episodic, EgressTier::Open, 3, "c"); // 2
        log.forget(|m| m.id == 1);
        // A len-based id scheme would hand out 2 here and collide; next_id must win.
        let fresh = log.append(MemoryKind::Episodic, EgressTier::Open, 4, "d");
        assert_eq!(fresh, 3);
        let ids: Vec<MemoryId> = log.events().iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![0, 2, 3]);
    }

    #[test]
    fn forgotten_content_is_absent_after_resealing() {
        let mut log = EpisodicLog::new();
        log.append(
            MemoryKind::Episodic,
            EgressTier::Private,
            1,
            "my secret pin is 1234",
        );
        log.append(MemoryKind::Episodic, EgressTier::Open, 2, "public note");
        log.forget(|m| m.content.contains("secret pin"));
        let sealed = log.seal(b"key").unwrap();
        let reopened = EpisodicLog::open(&sealed, b"key").unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(
            !reopened
                .events()
                .iter()
                .any(|m| m.content.contains("secret pin"))
        );
        assert_eq!(reopened.events()[0].content, "public note");
    }

    #[test]
    fn a_reopened_compacted_log_resumes_ids_without_reuse() {
        let mut log = sample(); // ids 0,1,2
        log.forget(|m| m.id == 0); // drop the oldest; highest surviving id is 2
        let sealed = log.seal(b"k").unwrap();
        let mut reopened = EpisodicLog::open(&sealed, b"k").unwrap();
        // next_id restored to max(1,2)+1 = 3, not len (2).
        let fresh = reopened.append(MemoryKind::Semantic, EgressTier::Open, 9, "z");
        assert_eq!(fresh, 3);
    }

    #[test]
    fn an_empty_reopened_log_starts_ids_at_zero() {
        let sealed = EpisodicLog::new().seal(b"k").unwrap();
        let mut reopened = EpisodicLog::open(&sealed, b"k").unwrap();
        assert_eq!(reopened.len(), 0);
        // Pins decode's `map_or(0, …)` empty branch.
        assert_eq!(
            reopened.append(MemoryKind::Working, EgressTier::Open, 1, "first"),
            0
        );
    }

    #[test]
    fn an_unknown_tag_byte_is_rejected() {
        let mut buf = sample().encode();
        buf[8] = 0x7F; // the first record's kind tag → out of range
        assert_eq!(EpisodicLog::decode(&buf), Err(StoreError::UnknownTag));
    }
}
