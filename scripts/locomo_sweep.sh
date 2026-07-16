#!/usr/bin/env bash
# Full-dataset fusion-weight sweep: runs benches/locomo_sweep in two 5-conversation halves (so
# candle's native-memory buildup can't reach across all 10 in one process) and aggregates the
# machine-readable per-config `SUM`/`N` lines into full-10 mean Recall@k. The paraphrase-R@5
# guardrail column is conversation-independent, so it is taken from the first half's own table.
#
#   LOCOMO_PATH=$PWD/locomo10.json bash scripts/locomo_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LOCOMO_PATH:?set LOCOMO_PATH to a downloaded data/locomo10.json}"

bin=$(cargo bench --bench locomo_sweep --features secure,local-embed --no-run 2>&1 \
        | grep -oE 'target[\\/].*locomo_sweep-[a-f0-9]+(\.exe)?' | head -1)
[ -n "$bin" ] || { echo "could not locate the built sweep binary" >&2; exit 1; }

a=$(mktemp); b=$(mktemp)
echo "half A: conversations 0..5"
SWEEP_START=0 SWEEP_CONVS=5 "./$bin" >"$a" 2>/dev/null
echo "half B: conversations 5..10"
SWEEP_START=5 SWEEP_CONVS=5 "./$bin" >"$b" 2>/dev/null

python - "$a" "$b" <<'PY'
import sys, re
a, b = sys.argv[1], sys.argv[2]
names = ["balanced  d1/r1/k1","semantic  d4/r1/k1","dense2    d2/r1/k1","dense3    d3/r1/k1",
         "dense6    d6/r1/k1","d4+kw2    d4/r1/k2","d3+kw2    d3/r1/k2","d2+kw2    d2/r1/k2",
         "d3+kw2-r0 d3/r0/k2"]
r5=[0.0]*len(names); r10=[0.0]*len(names); n=0
para={}
for f in (a,b):
    for line in open(f):
        m=re.match(r"SUM (\d+) (\S+) (\S+)", line)
        if m: i=int(m.group(1)); r5[i]+=float(m.group(2)); r10[i]+=float(m.group(3))
        m=re.match(r"N (\d+)", line)
        if m: n+=int(m.group(1))
# paraphrase-R@5 from half A's table rows (last float on each config row)
for line in open(a):
    for nm in names:
        key=nm.split()[0]
        if key in line and "/" in line:
            fl=re.findall(r"\d\.\d{3}", line)
            if len(fl)>=3: para[nm]=float(fl[-1])
order=sorted(range(len(names)), key=lambda i: -r5[i]/n)
print(f"\nFULL LoCoMo fusion-weight sweep — mean Recall@k over {n} answerable questions (all 10 conversations)\n")
print("  config                LoCoMo-R@5  LoCoMo-R@10  paraphrase-R@5")
sem=1
for i in order:
    mark = "  <- best LoCoMo R@5" if i==order[0] else ""
    print(f"  {names[i]}   {r5[i]/n:.3f}       {r10[i]/n:.3f}        {para.get(names[i],float('nan')):.3f}{mark}")
best=order[0]
print(f"\n  best vs semantic() default:  LoCoMo R@5 {(r5[best]-r5[sem])/n*100:+.1f} pts,  R@10 {(r10[best]-r10[sem])/n*100:+.1f} pts,  paraphrase R@5 {(para.get(names[best],0)-para.get(names[sem],0))*100:+.1f} pts")
PY
rm -f "$a" "$b"
