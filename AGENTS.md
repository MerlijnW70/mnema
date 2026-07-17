# AGENTS.md

Any AI agent working here follows one rule: **no change is done until every behavior it adds or touches is proven by a test.** A green build must *mean* something.

Before you consider a change done:

- `cargo fmt --all --check` is clean.
- `cargo clippy -p mnema-core --all-targets -- -D warnings` and `cargo clippy --features secure -- -D warnings` are clean.
- `cargo test -p mnema-core` and `cargo test --features secure` pass.
- `mnema-core` remains a strictly zero-dependency, `unsafe`-free leaf — `cargo tree -p mnema-core -e normal` must stay a single line.
