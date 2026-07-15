"""
Standard-retriever baselines on the LoCoMo evidence-retrieval task, replicating mnema's
benches/locomo.rs eval protocol EXACTLY so the numbers are apples-to-apples with mnema's
lexical (0.225/0.323) and semantic MiniLM (0.401/0.467).

Protocol (mirrors eval_range in benches/locomo.rs):
  * per conversation, build a fresh index over every turn of every session_* array,
    doc text = f"{speaker}: {text}", keyed by dia_id.
  * for each QA: parse gold evidence dia_ids (scan D<digit>..:.. tokens), keep only those
    actually stored; skip if none.
  * retrieve top-10 by the retriever's score; recall@5 = |gold in top5|/|gold|,
    recall@10 = |gold in top10|/|gold|. Mean over every answerable question across all 10 convs.
"""
import json, re, math, sys
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "locomo10.json"
data = json.load(open(PATH, encoding="utf-8"))

# ---- exact port of evidence_ids() from benches/locomo.rs -------------------------------
def evidence_ids(raw: str):
    cleaned = "".join(c if (c.isascii() and c.isalnum()) or c == ":" else " " for c in raw)
    out = []
    for t in cleaned.split():
        if t[0] == "D" and ":" in t and len(t) > 1 and t[1].isdigit():
            out.append(t)
    return out

# ---- standard tokenizer for lexical retrieval (lowercase alphanumeric words) ------------
TOK = re.compile(r"[a-z0-9]+")
def tok(s: str):
    return TOK.findall(s.lower())

# ---- Okapi BM25 (k1=1.5, b=0.75), fresh index per conversation -------------------------
def bm25_topk(docs_tokens, query_tokens, k1=1.5, b=0.75, k=10):
    N = len(docs_tokens)
    df = Counter()
    for d in docs_tokens:
        for w in set(d):
            df[w] += 1
    avgdl = sum(len(d) for d in docs_tokens) / N if N else 0.0
    idf = {w: math.log((N - n + 0.5) / (n + 0.5) + 1.0) for w, n in df.items()}
    tfs = [Counter(d) for d in docs_tokens]
    scores = [0.0] * N
    q = [w for w in query_tokens if w in idf]
    for i, (tf, d) in enumerate(zip(tfs, docs_tokens)):
        dl = len(d)
        s = 0.0
        for w in q:
            f = tf.get(w, 0)
            if f:
                s += idf[w] * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl))
        scores[i] = s
    order = sorted(range(N), key=lambda i: scores[i], reverse=True)
    return order[:k]

# ---- TF-IDF cosine (smooth idf, l2-normalized), fresh index per conversation -----------
def tfidf_topk(docs_tokens, query_tokens, k=10):
    N = len(docs_tokens)
    df = Counter()
    for d in docs_tokens:
        for w in set(d):
            df[w] += 1
    idf = {w: math.log((1 + N) / (1 + n)) + 1.0 for w, n in df.items()}
    def vec(tokens):
        c = Counter(tokens)
        v = {w: c[w] * idf[w] for w in c if w in idf}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / nrm for w, x in v.items()}
    qv = vec(query_tokens)
    scores = []
    for d in docs_tokens:
        dv = vec(d)
        # cosine = dot (both l2-normalized)
        common = qv.keys() & dv.keys()
        scores.append(sum(qv[w] * dv[w] for w in common))
    order = sorted(range(N), key=lambda i: scores[i], reverse=True)
    return order[:k]

def evaluate(retriever):
    r5_sum = r10_sum = 0.0
    questions = 0
    for sample in data:
        conv = sample["conversation"]
        # ingest turns in insertion order; dia_id -> doc index
        docs = []           # doc token lists
        idx_of = {}         # dia_id -> index
        for key, val in conv.items():
            if not (key.startswith("session_") and isinstance(val, list)):
                continue
            for turn in val:
                dia = turn.get("dia_id")
                text = turn.get("text")
                if dia is None or text is None:
                    continue
                speaker = turn.get("speaker", "")
                idx_of[dia] = len(docs)
                docs.append(tok(f"{speaker}: {text}"))
        for qa in sample.get("qa", []):
            question = qa.get("question")
            if question is None:
                continue
            ev = qa.get("evidence")
            gold_raw = ev if isinstance(ev, str) else json.dumps(ev)
            gold = [idx_of[d] for d in evidence_ids(gold_raw) if d in idx_of]
            if not gold:
                continue
            top = retriever(docs, tok(question), k=10)
            top5 = set(top[:5]); top10 = set(top[:10])
            denom = len(gold)
            r5_sum += sum(1 for g in gold if g in top5) / denom
            r10_sum += sum(1 for g in gold if g in top10) / denom
            questions += 1
    return r5_sum / questions, r10_sum / questions, questions

for name, fn in [("BM25 (Okapi k1=1.5,b=0.75)", bm25_topk), ("TF-IDF cosine", tfidf_topk)]:
    r5, r10, n = evaluate(fn)
    print(f"{name:32s} R@5={r5:.3f}  R@10={r10:.3f}  (n={n})")
