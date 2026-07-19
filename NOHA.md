# Noha — mnema's trust layer

mnema is audited by [**noha**](https://github.com/MerlijnW70/emerge), a language-agnostic engine
that proves the logic a change touches is *actually tested*. A green mnema build means: every
mutation of the covered logic is caught by a test, the zero-dependency core stays a pure leaf, and
the privacy invariants hold as declared constraints — not just as prose.

## What is enforced

Two config surfaces, one per crate:

| File | Command | What it certifies |
| --- | --- | --- |
| [`mnema-core/noha.yaml`](mnema-core/noha.yaml) | `noha gate` (run in `mnema-core/`) | The core is a **zero-dependency, unsafe-free leaf**: no non-std imports, `Cargo.toml` declares zero deps, and the pure-logic promise holds — `forbid ** std::fs / std::net / std::process / std::env` (the core computes, it never touches the world). Plus the fail-closed lint and full-surface check. |
| [`noha.yaml`](noha.yaml) (workspace root) | `noha prober` | **Behavioral mutation coverage** over both crates' logic, killed by the full suite (`--features secure,mcp,http-embed`). Plus architectural constraints: filesystem access confined to the keyfile module and the two binaries; process control to the binaries + keyfile (the Windows ACL shell-out); `ureq` and all `std::net` confined to the one opt-in embedder module. |

Run locally:

```sh
cd mnema-core && noha gate          # leaf certification
cd .. && noha prober                # full mutation coverage (slow; --diff <base> for PR-scoped)
noha report                         # the compliance manifest
```

CI runs both on every push/PR (the `noha` job in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):
`noha gate` on the core, and `noha prober --diff` on the lines a change touches, so no new survivor
lands with a PR.

## The mutation-coverage floor: 4 accepted survivors

The committed baseline (`.noha/baseline.tsv`) records **4 accepted survivors** — mutations no test
kills. Each is accepted for cause, not overlooked. All four are `#[cfg(unix)]` code, so they are not
even compiled on a Windows dev machine (where this baseline was generated); CI's Linux leg compiles
them.

| Site | Mutation | Why accepted |
| --- | --- | --- |
| `src/keyfile.rs` — `open_private_new` | `.write(true)→false` | **Equivalent on the happy path is not the issue — killed on Linux CI:** `create_new(true).write(false)` errors (`InvalidInput`), so the existing unix owner-only-permissions test fails. Not observable on Windows (different `#[cfg]` branch, which uses `icacls`). |
| `src/keyfile.rs` — `open_private_new` | `.create_new(true)→false` | Drops the symlink-refusal guarantee; behavior on the tested happy path is identical (the temp is unlinked first), so only a symlink-planting test distinguishes it. `#[cfg(unix)]`-only. |
| `src/keyfile.rs` — `sync_parent_dir` | `!empty→empty` filter | **Equivalent mutant.** The parent-dir fsync is *best-effort durability* (its result is `let _ =`-discarded by design); mutating the empty-path filter only makes `File::open("")` fail and get skipped — `Ok` either way, no observable behavior without OS fault injection. |
| `src/bin/mnema-server.rs` — `write_atomic` | `!empty→empty` filter | **Equivalent mutant**, same best-effort dir-fsync as above. |

Two are genuine **equivalent mutants** (best-effort fsync with no user-space-observable effect — the
correct disposition for mutation testing is to accept, never to contort code into killing them). Two
are `#[cfg(unix)]` `OpenOptions` flags that the Linux CI leg covers. This is the inherent shape of
cross-platform mutation testing: no single machine reaches zero, but the platforms together cover the
behavioral surface. `noha report` therefore reads **`hardened` / Noha-Green** against this floor.

Deliberately outside the probed surface (documented in each file's header): `src/model_embed.rs` (a
candle transformer forward pass — not deterministic logic a ratchet can pin; the pooling math it
rests on lives in `mnema-core/src/embed.rs` and *is* probed).
