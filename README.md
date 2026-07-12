# mnema

**A local, encrypted memory layer for AI agents.** Give your AI a permanent, private memory that
never forgets — and never leaks to the cloud.

**One-line install. Zero configuration.**

[![CI](https://github.com/MerlijnW70/mnema/actions/workflows/ci.yml/badge.svg)](https://github.com/MerlijnW70/mnema/actions)
[![release](https://img.shields.io/github/v/release/MerlijnW70/mnema)](https://github.com/MerlijnW70/mnema/releases/latest)
![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)

## Install

**macOS / Linux** — no toolchain, no build:

```bash
curl -fsSL https://raw.githubusercontent.com/MerlijnW70/mnema/main/install.sh | sh
```

**Windows** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/MerlijnW70/mnema/main/install.ps1 | iex
```

Other ways: **with Rust**, `cargo install --git https://github.com/MerlijnW70/mnema mnema --features mcp`
· or **download a binary** for your OS from the [latest release](https://github.com/MerlijnW70/mnema/releases/latest).

You get two commands: `mnema` (the CLI) and `mnema-mcp` (the server your AI editor talks to).

## Try it in 30 seconds

The store is created and encrypted on first use — no key, no setup:

```bash
mnema remember mind.store open "the user prefers TypeScript"
mnema recall   mind.store 5 "language preferences"
# → the user prefers TypeScript
```

## Give it to your AI

Add mnema to your MCP client (Cursor, Claude Desktop, Claude Code) — that's the whole setup:

```jsonc
{
  "mcpServers": {
    "mnema": {
      "command": "mnema-mcp",
      "env": { "MNEMA_PATH": "~/mnema.store" }
    }
  }
}
```

Your agent now has `remember`, `recall`, `recent`, and more. Everything stays on your disk, encrypted.
(If `mnema-mcp` isn't on your `PATH`, use the absolute path the installer printed.)

## Why mnema

- **Private by design** — everything lives on *your* disk, encrypted. Memories you mark private are
  structurally blocked from ever reaching a cloud model. Nothing phones home.
- **Trustworthy** — built in Rust, 100% mutation-tested, and fail-closed: a wrong key or a crash mid-write
  never loses or leaks your memory.
- **Tiny & fast** — a ~0.4 MB binary with zero runtime dependencies. No Python, no database, no daemon.

## Commands

| Command | Action |
|---|---|
| `mnema remember <store> <tier> <text>` | Store a memory (`tier` = `open` / `redacted` / `private`) |
| `mnema recall <store> <k> <query>` | Retrieve the most relevant memories |
| `mnema fact <store> <subject> <attribute> <value>` | Store a belief (a newer value supersedes an older one) |
| `mnema stats <store>` | Memory health — counts by privacy tier |
| `mnema prune <store> <half_life> <threshold>` | Forget faded memories |
| `mnema keygen` | Print a strong passphrase to use as `MNEMA_KEY` |
| `mnema rekey <store>` | Re-encrypt the store under a fresh key |

Over MCP your agent also gets **recent**, **beliefs**, **reinforce**, and **forget**.

## Keys

You don't have to manage a key: omit `MNEMA_KEY` and mnema generates a random key file
(`<store>.key`) next to your store. Want a portable passphrase instead (a shared store, CI, an
env-only secret)? Set `MNEMA_KEY` to any string — or a strong random one:

```bash
export MNEMA_KEY="$(mnema keygen)"
```

## Troubleshooting

- **`cannot open store (wrong key or corrupt)`** — the store was sealed under a different key. Set
  `MNEMA_KEY` to the right passphrase, or make sure `<store>.key` is still next to the store.
- **Lost your `MNEMA_KEY`?** If you used a passphrase and lost it, encrypted data can't be recovered
  — that's the point. If you used the default key file, keep `<store>.key` safe; it *is* the key.
- **`already in use by another mnema process`** — one writer at a time. Close the other server or CLI
  and retry.
- **Server won't start after an interrupted `rekey`** — set `MNEMA_KEY` to the **old** passphrase and
  run `mnema rekey <store>` again to finish the migration.

## Under the hood

Curious how the privacy wall, encryption, retrieval, and the mutation-testing that proves it all
actually work? → **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## License

Dual-licensed under **MIT OR Apache-2.0**.
