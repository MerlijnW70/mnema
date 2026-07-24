//! The umbrella crate's capability confinement, mechanically held.
//!
//! mnema's privacy claims rest on *where* dangerous capabilities may appear, not merely on
//! whether they are imported:
//!
//! * filesystem access stays in the keyfile module and the two binaries — the library core
//!   (facade, store, http_embed) does no file I/O;
//! * process control stays in the binaries and the keyfile module (the Windows owner-only ACL
//!   restriction shells out to `icacls`);
//! * outbound HTTP stays in the one opt-in embedder module;
//! * raw socket access is likewise confined to that module — "never leaks to the cloud"
//!   starts here.
//!
//! Each rule below names the files that may hold the capability. Every other source in the
//! crate must be free of it, by any route: a `use`, a braced group, an alias, or a
//! fully-qualified call written inline. The last of these is the one an import-only check
//! cannot see, and it is exactly how this crate already writes `std::fs::rename` and
//! `ureq::AgentBuilder` — so a check that reads imports would report these rules as holding
//! vacuously while the capabilities are in active use.
//!
//! Sources are discovered from the directory, so a new file is confined the moment it exists.

#[path = "../mnema-core/tests/common/mod.rs"]
mod common;

use common::capability_hits;

/// One confinement rule: a capability, the files permitted to hold it, and the promise it
/// encodes.
struct Confine {
    /// The crate root the capability lives under (`"std"`, `"ureq"`).
    root: &'static str,
    /// The segment beneath it, or `None` to confine the whole crate (third-party deps).
    segment: Option<&'static str>,
    /// Paths, relative to the crate root, that may hold this capability.
    allowed: &'static [&'static str],
    /// The promise this rule keeps, quoted in the failure message.
    promise: &'static str,
}

const RULES: &[Confine] = &[
    Confine {
        root: "std",
        segment: Some("fs"),
        // `model_embed.rs` reads the Hugging Face cache (config, tokenizer, weights) when the
        // opt-in `local-embed` feature is on. That is real, intended file I/O — it is listed
        // here because a capability the code holds must be visible in the rule, not left to a
        // check that cannot see fully-qualified calls. It reads model assets only; the store,
        // the facade and the key material stay out of it.
        allowed: &[
            "src/keyfile.rs",
            "src/bin/mnema-server.rs",
            "src/bin/mnema.rs",
            "src/model_embed.rs",
        ],
        promise: "the library core — facade, store, http_embed — does no file I/O",
    },
    Confine {
        root: "std",
        segment: Some("process"),
        allowed: &[
            "src/bin/mnema-server.rs",
            "src/bin/mnema.rs",
            "src/keyfile.rs",
        ],
        promise: "process control stays in the binaries and the keyfile module",
    },
    Confine {
        root: "ureq",
        segment: None,
        allowed: &["src/http_embed.rs"],
        promise: "outbound HTTP stays in the one opt-in embedder module",
    },
    Confine {
        root: "std",
        segment: Some("net"),
        allowed: &["src/http_embed.rs"],
        promise: "no socket may be opened outside the opt-in embedder — \
                  'never leaks to the cloud' starts here",
    },
];

/// Every `.rs` file under the umbrella's `src/`, discovered rather than listed, keyed by a
/// forward-slash path relative to the crate root so the rules read the same on every platform.
fn umbrella_sources() -> Vec<(String, String)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(d) = stack.pop() {
        let entries =
            std::fs::read_dir(&d).unwrap_or_else(|e| panic!("cannot read {}: {e}", d.display()));
        for entry in entries {
            let path = entry.expect("readable dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let rel = path
                    .strip_prefix(root)
                    .expect("source lives under the crate root")
                    .to_string_lossy()
                    .replace('\\', "/");
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
                found.push((rel, text));
            }
        }
    }
    assert!(
        !found.is_empty(),
        "no umbrella sources discovered — the test is not looking where it thinks"
    );
    found
}

/// No source outside a rule's allowlist reaches that rule's capability.
#[test]
fn capabilities_stay_where_they_are_confined() {
    let sources = umbrella_sources();
    let mut violations = Vec::new();

    for rule in RULES {
        for (path, text) in &sources {
            if rule.allowed.contains(&path.as_str()) {
                continue;
            }
            for hit in capability_hits(text, rule.root, rule.segment) {
                violations.push(format!("{path}: {hit}\n      breaks: {}", rule.promise));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "capability confinement broken:\n  {}",
        violations.join("\n  ")
    );
}

/// Every crate root in the workspace bans `unsafe`.
///
/// `#![forbid(unsafe_code)]` applies to the crate root it sits in and nowhere else. Each binary
/// is its own crate root, so the library's ban does not reach `src/bin/*` — those files were
/// outside it until this test existed, including the MCP server that parses untrusted
/// JSON-RPC. The attribute is one line, and one line is easy to lose in a refactor; this is
/// what keeps it.
#[test]
fn every_crate_root_forbids_unsafe() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let roots = [
        "src/lib.rs",
        "src/bin/mnema.rs",
        "src/bin/mnema-server.rs",
        "mnema-core/src/lib.rs",
    ];

    let mut unbanned = Vec::new();
    for rel in roots {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        // The attribute must be real code, not a mention in a doc comment explaining it.
        let code = common::strip_comments_and_literals(&text);
        if !code.replace(' ', "").contains("#![forbid(unsafe_code)]") {
            unbanned.push(rel);
        }
    }

    assert!(
        unbanned.is_empty(),
        "crate roots without `#![forbid(unsafe_code)]`: {unbanned:?}"
    );
}

/// A confinement rule whose allowlist names a file that no longer exists is silently dead —
/// it would keep passing while confining nothing. Catch the drift at the rule, not months later.
#[test]
fn every_allowlisted_path_exists() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut missing = Vec::new();
    for rule in RULES {
        for allowed in rule.allowed {
            if !root.join(allowed).exists() {
                missing.push(format!(
                    "{allowed} is allowed to hold `{}{}` but does not exist",
                    rule.root,
                    rule.segment.map(|s| format!("::{s}")).unwrap_or_default()
                ));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "confinement rules reference files that are gone:\n  {}",
        missing.join("\n  ")
    );
}

/// Proof the rules bite. For each rule, plant its capability in a source that is *not*
/// allowlisted and require the scan to catch it — otherwise a green
/// `capabilities_stay_where_they_are_confined` proves nothing.
#[test]
fn every_rule_is_enforced_not_vacuous() {
    for rule in RULES {
        let capability = match rule.segment {
            Some(seg) => format!("{}::{seg}::something()", rule.root),
            None => format!("{}::something()", rule.root),
        };
        let planted = format!("fn _canary() {{ let _ = {capability}; }}\n");
        assert!(
            !capability_hits(&planted, rule.root, rule.segment).is_empty(),
            "the scan cannot detect `{}{}` at all — the rule confining it to {:?} is decorative",
            rule.root,
            rule.segment.map(|s| format!("::{s}")).unwrap_or_default(),
            rule.allowed
        );
    }
}

/// The allowlists are meant to describe reality. A capability that no allowlisted file
/// actually holds means the rule has drifted from the code — either the capability moved and
/// the rule was not updated, or the rule was aspirational to begin with. Either way the
/// confinement is not saying what it appears to say.
#[test]
fn each_rule_confines_a_capability_that_is_actually_used() {
    let sources = umbrella_sources();
    let mut inert = Vec::new();

    for rule in RULES {
        let used_somewhere = sources.iter().any(|(path, text)| {
            rule.allowed.contains(&path.as_str())
                && !capability_hits(text, rule.root, rule.segment).is_empty()
        });
        if !used_somewhere {
            inert.push(format!(
                "`{}{}` is confined to {:?} but appears in none of them",
                rule.root,
                rule.segment.map(|s| format!("::{s}")).unwrap_or_default(),
                rule.allowed
            ));
        }
    }

    assert!(
        inert.is_empty(),
        "confinement rules have drifted from the code:\n  {}",
        inert.join("\n  ")
    );
}
