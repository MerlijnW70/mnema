# noha Integrity SOP — Autonomous Integrity Loop

This **rust** project is guarded by noha-core. Any AI agent working here must treat the
Autonomous Integrity Loop as the highest priority: a green build must *mean* something.

## The golden rule

No commit or PR is finished unless the codebase is 100% hardened (0 survivors).

## The loop — run on every change

1. **Gate** — run this project's standard guard: `cargo clippy --all-targets -- -D warnings`. If it is red, stop and fix it. (`noha gate`
   additionally applies **only** to a zero-dependency Rust leaf crate — it audits `./src` imports and
   `Cargo.toml` declarations; for every other stack, the prober loop below IS the integrity gate.)
2. **Probe** — `noha prober --diff` (`noha_prober_diff`). If survivors remain, go to step 3.
3. **Repair** — for each survivor call `noha_repair` (or `noha prober --explain`); read the
   Repair-Request v2 and write the test it describes (its `witness` + `assert`, at `test_location`).
4. **Verify** — run the prober again; the survivor count must reach 0.
5. **Finalize** — commit only when the prober reports 0 survivors, then `noha prober --update`.

Test command: `cargo test --quiet`  ·  Standard guard: `cargo clippy --all-targets -- -D warnings`

## Tools (your source of truth)

`noha_gate` · `noha_prober_diff` · `noha_prober_explain` · `noha_repair`

Trust proof, not prompts: the prober re-run is the evidence — never a passing-tests hunch.
