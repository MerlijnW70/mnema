//! The core's central promise, mechanically held: **it computes, it never touches the world.**
//!
//! `mnema-core` carries the egress-privacy invariant and all pure logic. Nothing in it may
//! reach the filesystem, the network, the process table, or the environment — not through an
//! import, and not through a fully-qualified call written inline.
//!
//! An import-only check does not hold this promise. `std::fs::read(path)` needs no `use`, so a
//! checker that reads `use` declarations reports "no source imports std::fs" while the code
//! reads the disk. This test closes that gap: it strips comments and literals, then looks for
//! every route to the capability (see `common::capability_hits` for why that set is complete).
//!
//! Sources are discovered from the directory rather than listed, so a file added to `src/`
//! is covered the moment it exists — there is no list to forget to update.

mod common;

use common::capability_hits;

/// The capabilities the core may never reach, and why each one matters.
///
/// `std::os` is included beyond the four the crate documents: it is a back door to the same
/// filesystem (`std::os::unix::fs::OpenOptionsExt` and friends) and would otherwise satisfy a
/// `std::fs` check while still touching the disk.
const FORBIDDEN: &[(&str, &str)] = &[
    ("fs", "the core must not read or write the filesystem"),
    (
        "net",
        "the core must not open a socket — 'never leaks to the cloud' starts here",
    ),
    ("process", "the core must not spawn or control processes"),
    ("env", "the core must not read ambient environment state"),
    (
        "os",
        "platform extension traits are a back door to the filesystem",
    ),
];

/// Every `.rs` file under the crate's `src/`, discovered rather than listed.
fn core_sources() -> Vec<(String, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        let entries =
            std::fs::read_dir(&d).unwrap_or_else(|e| panic!("cannot read {}: {e}", d.display()));
        for entry in entries {
            let path = entry.expect("readable dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
                found.push((path.display().to_string(), text));
            }
        }
    }
    assert!(
        !found.is_empty(),
        "no core sources discovered — the test is not looking where it thinks"
    );
    found
}

/// The core reaches no forbidden capability, by any route, in any of its sources.
#[test]
fn the_core_touches_nothing() {
    let sources = core_sources();
    let mut violations = Vec::new();

    for (path, text) in &sources {
        for (segment, why) in FORBIDDEN {
            for hit in capability_hits(text, "std", Some(segment)) {
                violations.push(format!("{path}: {hit} — {why}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "the zero-dependency core reached the outside world:\n  {}",
        violations.join("\n  ")
    );
}

/// The check above is only as good as its reach. If `src/` were empty, mis-pathed, or the
/// scanner silently returned nothing, `the_core_touches_nothing` would pass vacuously — the
/// most dangerous kind of green. This test proves the scanner bites on the real sources by
/// planting a violation in a copy of each one and requiring it to be caught.
#[test]
fn the_check_is_not_vacuous() {
    for (path, text) in core_sources() {
        let planted =
            format!("{text}\nfn _canary() {{ let _ = std::fs::read(\"/etc/passwd\"); }}\n");
        assert!(
            !capability_hits(&planted, "std", Some("fs")).is_empty(),
            "the scanner failed to catch a planted filesystem call in {path} — \
             a green result from `the_core_touches_nothing` would be meaningless"
        );
    }
}
