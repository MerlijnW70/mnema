#!/usr/bin/env bash
# Channel B for Mnema's exact vector recall (docs/SELF-EVOLUTION.md, Part 25): the
# O(N) search latency an HNSW/ANN index would later optimize, one JSON line. Point the
# driver at it with EVOLVE_FITNESS=scripts/fitness-mnema.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo bench --bench mnema --quiet
