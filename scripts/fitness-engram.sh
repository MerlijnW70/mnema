#!/usr/bin/env bash
# Channel B for Engram's exact vector recall (docs/SELF-EVOLUTION.md, Part 25): the
# O(N) search latency an HNSW/ANN index would later optimize, one JSON line. Point the
# driver at it with EVOLVE_FITNESS=scripts/fitness-engram.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo bench --bench engram --quiet
