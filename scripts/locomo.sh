#!/usr/bin/env bash
# Full LoCoMo retrieval benchmark for mnema — robust to candle's native-memory buildup on long
# runs by evaluating ONE conversation per fresh subprocess (with retry), then aggregating.
#
# Needs the dataset:
#   curl -sL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o locomo10.json
#   LOCOMO_PATH=$PWD/locomo10.json bash scripts/locomo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LOCOMO_PATH:?set LOCOMO_PATH to a downloaded data/locomo10.json}"

echo "building the locomo bench (secure,local-embed)…"
bin=$(cargo +1.96 bench --bench locomo --features secure,local-embed --no-run 2>&1 \
        | grep -oE 'target[\\/].*locomo-[a-f0-9]+(\.exe)?' | head -1)
[ -n "$bin" ] || { echo "could not locate the built bench binary" >&2; exit 1; }

r5=0; r10=0; n=0
for i in $(seq 0 9); do
  ok=""
  for attempt in 1 2 3 4; do
    out=$(LOCOMO_ONLY="$i" "./$bin" 2>/dev/null | grep '^SEM' || true)
    if [ -n "$out" ]; then
      read -r _ a b c <<<"$out"
      r5=$(awk "BEGIN{print $r5+$a}"); r10=$(awk "BEGIN{print $r10+$b}"); n=$(awk "BEGIN{print $n+$c}")
      printf '  conversation %d: n=%s\n' "$i" "$c"; ok=1; break
    fi
    printf '  conversation %d: crash, retry %d\n' "$i" "$attempt"
  done
  [ -n "$ok" ] || { echo "conversation $i failed after retries" >&2; exit 1; }
done

awk "BEGIN{printf \"\nSEMANTIC full LoCoMo: R@5=%.3f  R@10=%.3f  (n=%d questions)\n\", $r5/$n, $r10/$n, $n}"
