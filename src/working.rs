//! Working memory — the fourth memory type (`docs/proposals/mnema-memory-layer.md`
//! §3.1). An ephemeral scratchpad for the current session: short-lived notes that
//! **expire with age** (a time-to-live horizon) and are **capacity-bounded** (the
//! oldest is evicted when full). Unlike the episodic/semantic stores it is not
//! persisted — it is the model's short-term working set, not its long-term record.
//!
//! Pure safe Rust, zero dependencies (ADR-0007 holds). The two load-bearing,
//! internal-tool-pinned rules are: a note is live iff its age is within the horizon (the
//! `<=` boundary is inclusive), and the store never holds more than `capacity`
//! notes (the oldest goes first).

/// One scratchpad note, stamped with the logical time it was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    pub at: u64,
    pub content: String,
}

/// A bounded, self-expiring scratchpad. `at`/`now` are caller-supplied logical times
/// (Mnema never reads the wall clock), keeping it deterministic and testable.
#[derive(Clone, Debug)]
pub struct WorkingMemory {
    horizon: u64,
    capacity: usize,
    notes: Vec<Note>,
}

impl WorkingMemory {
    /// A scratchpad that keeps notes for `horizon` ticks of age and at most
    /// `capacity` of them at once.
    pub fn new(horizon: u64, capacity: usize) -> Self {
        Self {
            horizon,
            capacity,
            notes: Vec::new(),
        }
    }

    /// Write a note observed at logical time `at`. If this pushes the store past
    /// `capacity`, the note with the smallest `at` (the oldest *by time*) is evicted.
    /// This is time-based, not position-based, so an out-of-order write never evicts a
    /// genuinely newer note — `at` is caller-supplied and not required to be monotonic.
    pub fn note(&mut self, at: u64, content: impl Into<String>) {
        self.notes.push(Note {
            at,
            content: content.into(),
        });
        if self.notes.len() > self.capacity {
            // Evict the oldest by timestamp; on a tie the earliest-inserted such note goes
            // (`min_by_key` keeps the first minimum), matching the append-order behaviour
            // when timestamps are monotonic.
            if let Some(oldest) = self
                .notes
                .iter()
                .enumerate()
                .min_by_key(|(_, n)| n.at)
                .map(|(i, _)| i)
            {
                self.notes.remove(oldest);
            }
        }
    }

    /// Whether a note of age `now - at` is still within the horizon (inclusive).
    fn is_live(&self, at: u64, now: u64) -> bool {
        now.saturating_sub(at) <= self.horizon
    }

    /// The notes still live at `now`, newest first (by `at`), without mutating. Ordering
    /// is by timestamp — not insertion position — so an out-of-order write still sorts
    /// correctly; equal timestamps keep insertion order (the sort is stable).
    pub fn active(&self, now: u64) -> Vec<&Note> {
        let mut live: Vec<&Note> = self
            .notes
            .iter()
            .filter(|n| self.is_live(n.at, now))
            .collect();
        live.sort_by_key(|n| std::cmp::Reverse(n.at)); // newest first, stable on ties
        live
    }

    /// Permanently drop every note that has expired as of `now`.
    pub fn prune(&mut self, now: u64) {
        self.notes
            .retain(|n| now.saturating_sub(n.at) <= self.horizon);
    }

    /// Number of notes currently held (live or not-yet-pruned).
    pub fn len(&self) -> usize {
        self.notes.len()
    }

    /// Whether the scratchpad holds no notes.
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.notes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contents<'a>(notes: &[&'a Note]) -> Vec<&'a str> {
        notes.iter().map(|n| n.content.as_str()).collect()
    }

    #[test]
    fn active_returns_live_notes_newest_first() {
        let mut w = WorkingMemory::new(100, 10);
        w.note(1, "a");
        w.note(2, "b");
        w.note(3, "c");
        assert_eq!(contents(&w.active(3)), vec!["c", "b", "a"]);
    }

    #[test]
    fn the_horizon_boundary_is_inclusive() {
        let mut w = WorkingMemory::new(10, 10);
        w.note(0, "x");
        // age exactly == horizon (10) is still live; one tick later it is not.
        assert_eq!(contents(&w.active(10)), vec!["x"]);
        assert!(w.active(11).is_empty());
    }

    #[test]
    fn expired_notes_drop_out_of_active() {
        let mut w = WorkingMemory::new(5, 10);
        w.note(1, "old");
        w.note(10, "new");
        // At now=12: "old" is age 11 (>5, gone), "new" is age 2 (live).
        assert_eq!(contents(&w.active(12)), vec!["new"]);
    }

    #[test]
    fn capacity_evicts_the_oldest_note() {
        let mut w = WorkingMemory::new(100, 2);
        w.note(1, "a");
        w.note(2, "b");
        w.note(3, "c"); // pushes past capacity 2 → "a" evicted
        assert_eq!(w.len(), 2);
        assert_eq!(contents(&w.active(3)), vec!["c", "b"]);
    }

    #[test]
    fn eviction_and_ordering_are_by_time_not_insertion_order() {
        // `at` is caller-supplied and need not be monotonic. With an out-of-order write,
        // the genuinely oldest note (by time) must be evicted — not merely the first
        // inserted — and `active` must still order newest-first by timestamp.
        let mut w = WorkingMemory::new(100, 2);
        w.note(10, "ten");
        w.note(20, "twenty");
        w.note(5, "five"); // over capacity: the oldest BY TIME is "five"@5 — but so far
        // len is 3 > 2, so one is evicted. The oldest by time among {10,20,5} is 5.
        assert_eq!(w.len(), 2);
        // "five"@5 is evicted (position-based eviction would have wrongly dropped "ten"@10).
        let live = contents(&w.active(100));
        assert_eq!(live, vec!["twenty", "ten"]); // newest-first by `at`, not by insertion
    }

    #[test]
    fn prune_permanently_drops_expired_notes() {
        let mut w = WorkingMemory::new(5, 10);
        w.note(1, "old");
        w.note(10, "new");
        w.prune(10); // "old" is age 9 (>5) → gone; "new" is age 0 → kept
        assert_eq!(w.len(), 1);
        assert_eq!(contents(&w.active(10)), vec!["new"]);
    }

    #[test]
    fn prune_keeps_a_note_at_exactly_the_horizon() {
        // Pins prune's own `<=` boundary: a note whose age equals the horizon survives.
        let mut w = WorkingMemory::new(5, 10);
        w.note(5, "edge"); // at now=10 → age 5 == horizon → kept
        w.note(3, "over"); // age 7 > horizon → dropped
        w.prune(10);
        assert_eq!(w.len(), 1);
        assert_eq!(contents(&w.active(10)), vec!["edge"]);
    }

    #[test]
    fn clear_empties_the_scratchpad() {
        let mut w = WorkingMemory::new(10, 10);
        w.note(1, "a");
        assert!(!w.is_empty());
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }
}
