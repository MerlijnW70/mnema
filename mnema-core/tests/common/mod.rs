//! A source scanner for architectural capability checks, shared by the workspace's
//! architecture tests.
//!
//! # Why this exists
//!
//! mnema makes capability promises — "the core computes, it never touches the world",
//! "all outbound HTTP lives in one opt-in module". A check that only inspects `use`
//! declarations does not hold those promises: `std::fs::read(path)` written inline reaches
//! the filesystem without ever naming an import. This scanner closes that gap.
//!
//! # Why it is sound
//!
//! Any path from this workspace's own source to a `std` capability must, somewhere in the
//! scanned text, either
//!
//! 1. spell the path out — `std::fs::read`, `::std::fs::read`, `use std::fs as anything`; or
//! 2. name the segment inside a braced group — `use std::{fs, io}`.
//!
//! A re-export (`pub use std::fs;`) is itself case 1 at the re-export site, and an alias
//! (`use std::fs as f;`) is case 1 at the `use`. So scanning for both forms across every
//! source file of a crate covers every route to the capability. The one construct that
//! would evade it is a glob (`use std::*;`), which is reported separately.
//!
//! Comments and literals are removed before scanning, so prose that merely *mentions* a
//! capability — this module's own doc header, or `keyfile.rs`'s explanation of why it holds
//! the filesystem — is not a violation. Only code counts.

/// Replace every comment and string/char literal with spaces, preserving byte length so
/// reported offsets still line up with the original text.
///
/// Handles line comments, nested block comments, plain and raw strings (`r"…"`, `r#"…"#`),
/// char literals, and backslash escapes — the full set Rust admits, because a scanner that
/// mishandles any of them either misses a violation or invents one.
pub fn strip_comments_and_literals(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = vec![b' '; b.len()];
    let mut i = 0;

    // Keep newlines so line numbers survive.
    let keep_newlines = |out: &mut Vec<u8>, from: usize, to: usize| {
        for (k, item) in b[from..to].iter().enumerate() {
            if *item == b'\n' {
                out[from + k] = b'\n';
            }
        }
    };

    while i < b.len() {
        // Line comment.
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            let start = i;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            keep_newlines(&mut out, start, i);
            continue;
        }
        // Block comment (nested, as Rust allows).
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let start = i;
            let mut depth = 1usize;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            keep_newlines(&mut out, start, i);
            continue;
        }
        // Raw string: r"…", r#"…"#, r##"…"##, …
        if b[i] == b'r' && i + 1 < b.len() && (b[i + 1] == b'"' || b[i + 1] == b'#') {
            let mut j = i + 1;
            let mut hashes = 0usize;
            while j < b.len() && b[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < b.len() && b[j] == b'"' {
                let start = i;
                j += 1;
                loop {
                    if j >= b.len() {
                        break;
                    }
                    if b[j] == b'"' {
                        let mut k = j + 1;
                        let mut seen = 0usize;
                        while k < b.len() && b[k] == b'#' && seen < hashes {
                            seen += 1;
                            k += 1;
                        }
                        if seen == hashes {
                            j = k;
                            break;
                        }
                    }
                    j += 1;
                }
                keep_newlines(&mut out, start, j.min(b.len()));
                i = j;
                continue;
            }
        }
        // Plain string.
        if b[i] == b'"' {
            let start = i;
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if b[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            keep_newlines(&mut out, start, i.min(b.len()));
            continue;
        }
        // Char literal — distinguished from a lifetime (`'a`) by the closing quote.
        if b[i] == b'\'' {
            let mut j = i + 1;
            if j < b.len() && b[j] == b'\\' {
                j += 2;
            } else if j < b.len() {
                j += 1;
            }
            if j < b.len() && b[j] == b'\'' {
                i = j + 1;
                continue; // already blanked
            }
            // A lifetime: copy the tick through and carry on.
            out[i] = b'\'';
            i += 1;
            continue;
        }
        out[i] = b[i];
        i += 1;
    }

    String::from_utf8_lossy(&out).into_owned()
}

/// Remove every ASCII whitespace byte, so `std :: fs` and `std::fs` compare equal.
fn squeeze(src: &str) -> String {
    src.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Every way `crate_or_std::segment` can be reached from `src`, as a list of human-readable
/// reasons. Empty means the capability is absent.
///
/// `root` is the crate root (`"std"`, `"ureq"`, …) and `segment` the capability beneath it
/// (`"fs"`, `"net"`, …). Pass `segment = None` to check for the root itself, which is how a
/// third-party crate like `ureq` is detected.
pub fn capability_hits(src: &str, root: &str, segment: Option<&str>) -> Vec<String> {
    let code = squeeze(&strip_comments_and_literals(src));
    let mut hits = Vec::new();

    let full = match segment {
        Some(seg) => format!("{root}::{seg}"),
        None => format!("{root}::"),
    };
    if code.contains(&full) {
        hits.push(format!("names `{full}` directly"));
    }

    // Braced group: `use std::{fs, io}` reaches fs without spelling `std::fs`.
    if let Some(seg) = segment {
        let opener = format!("{root}::{{");
        let mut from = 0usize;
        while let Some(rel) = code[from..].find(&opener) {
            let start = from + rel + opener.len();
            let mut depth = 1usize;
            let mut end = start;
            for (k, c) in code[start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + k;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let group = &code[start..end];
            if group
                .split(',')
                .any(|item| item == seg || item.starts_with(&format!("{seg}::")))
            {
                hits.push(format!("reaches `{seg}` via a `{root}::{{…}}` group"));
            }
            from = end.max(start + 1);
        }
    }

    // A glob is the one construct the checks above cannot see through.
    if code.contains(&format!("{root}::*")) {
        hits.push(format!(
            "glob-imports `{root}::*`, which hides what it pulls in"
        ));
    }

    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_mentioning_a_capability_is_not_a_violation() {
        let src = "//! we deliberately keep std::fs out of here\nfn f() {}\n";
        assert!(capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_string_literal_mentioning_a_capability_is_not_a_violation() {
        let src = "fn f() { let s = \"std::fs::read\"; }";
        assert!(capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_raw_string_mentioning_a_capability_is_not_a_violation() {
        let src = "fn f() { let s = r#\"std::fs::read\"#; }";
        assert!(capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_block_comment_mentioning_a_capability_is_not_a_violation() {
        let src = "/* std::fs::read /* nested */ still a comment */ fn f() {}";
        assert!(capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_fully_qualified_call_is_caught() {
        let src = "fn f() { let _ = std::fs::read(\"/etc/passwd\"); }";
        assert_eq!(capability_hits(src, "std", Some("fs")).len(), 1);
    }

    #[test]
    fn whitespace_between_path_segments_does_not_hide_it() {
        let src = "fn f() { let _ = std :: fs :: read(\"x\"); }";
        assert_eq!(capability_hits(src, "std", Some("fs")).len(), 1);
    }

    #[test]
    fn a_plain_use_is_caught() {
        let src = "use std::fs;\nfn f() { let _ = fs::read(\"x\"); }";
        assert!(!capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn an_aliased_use_is_caught() {
        let src = "use std::fs as disguise;\nfn f() { let _ = disguise::read(\"x\"); }";
        assert!(!capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_braced_group_is_caught() {
        let src = "use std::{fs, io};\nfn f() { let _ = fs::read(\"x\"); }";
        assert!(!capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_nested_braced_group_is_caught() {
        let src = "use std::{io::Write, fs::File};\n";
        assert!(!capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_sibling_segment_is_not_a_false_positive() {
        let src = "use std::{io, cmp::Ordering};\nfn f() {}";
        assert!(capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn os_specific_fs_extensions_are_not_confused_with_std_fs() {
        // `std::os::unix::fs` is a different path; it must not register as `std::fs`.
        let src = "use std::os::unix::fs::OpenOptionsExt;\n";
        assert!(capability_hits(src, "std", Some("fs")).is_empty());
        assert!(!capability_hits(src, "std", Some("os")).is_empty());
    }

    #[test]
    fn a_glob_is_reported() {
        let src = "use std::*;\n";
        assert!(!capability_hits(src, "std", Some("fs")).is_empty());
    }

    #[test]
    fn a_bare_crate_root_is_detected_without_a_segment() {
        let src = "fn f() { let a = ureq::AgentBuilder::new(); }";
        assert!(!capability_hits(src, "ureq", None).is_empty());
    }

    #[test]
    fn line_numbers_survive_stripping() {
        let src = "// one\n/* two\nstill two */\nfn f() {}\n";
        let stripped = strip_comments_and_literals(src);
        assert_eq!(stripped.lines().count(), src.lines().count());
        assert_eq!(stripped.len(), src.len());
    }
}
