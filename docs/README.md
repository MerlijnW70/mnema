# mnema — documentation

| File | What it is |
|---|---|
| [`DESIGN.md`](DESIGN.md) | The full design: architecture, the four memory types, the write/read paths, the threat model, and the phased build plan. Start here. |
| [`NARRATIVE.md`](NARRATIVE.md) | The story of building the layer under a mutation ratchet — why it was the test, the invariants pinned, the teeth-gap moments. |
| [`adr/0020-mnema-dependency-budget.md`](adr/0020-mnema-dependency-budget.md) | Why a small, vetted, safe-Rust crypto set is allowed behind an optional `secure` feature (the core stays zero-dependency). |
| [`adr/0021-mnema-egress-tier.md`](adr/0021-mnema-egress-tier.md) | The per-memory egress tier — how a `Private` memory is made *structurally* unable to reach a remote model. |
| [`adr/0022-memory-augmented-self-evolution.md`](adr/0022-memory-augmented-self-evolution.md) | An example *consumer*: giving a self-evolving agent a memory of its own history. Included because it shows the layer used in anger. |
| [`adr/0023-embedder-width-single-source.md`](adr/0023-embedder-width-single-source.md) | Why the default embedder's width is pinned as one library constant — a bug the ratchet structurally can't catch (recall-only drift between two binaries over a shared store). |

## A note on cross-references

`mnema` was extracted from the **emerge** research project, where it was built under the **noha**
mutation-coverage gate. The ADRs and narrative therefore cite records that live in *that* repository:

- **`ADR-0007`**, **`ADR-0012`**, **`ADR-0006`**, **`ADR-0014`** — emerge decisions (the
  `forbid(unsafe)` wall, "own your hasher", the anti-reroll ledger, Repair-Request feedback).
- **`BND-*`** (e.g. `BND-performance-blindness`, `BND-statistical-quality`, `BND-tested-not-correct`)
  — emerge's first-class *boundaries*: the limits a behavioral gate structurally cannot cross.
- **`Part N`** — chapters of emerge's `docs/SELF-EVOLUTION.md` narrative.

These are kept verbatim rather than rewritten, so the reasoning is faithful to how it was actually
decided. To follow a reference, see the [emerge repository](https://github.com/MerlijnW70/emerge).
The two claims those references most often support:

- **"green means proven"** — noha kills a mutation only when a test's pass/fail flips, so a green
  ratchet means the changed logic is genuinely tested, not merely executed.
- **the performance-blindness boundary** — a *behavioral* gate cannot see a change that alters only
  speed or recall; those are measured on a benchmark (channel B), never pinned by the ratchet. This is
  why exact retrieval is the pinned oracle and the approximate `IvfIndex` is benchmarked, not pinned.
