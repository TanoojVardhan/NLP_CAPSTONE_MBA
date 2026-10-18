"""Reservoir-sample reviews from the old Amazon Electronics_5.json (one JSON object per line) for out-of-domain testing."""
import json, random, sys, csv, os
SRC, OUT, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
random.seed(42)
res, n = [], 0
with open(SRC, encoding="utf-8") as f:
    for line in f:
        n += 1
        if n % 5: continue            # look at every 5th line (still ~1.4M candidates), keeps the pass fast
        if len(res) < K:
            res.append(line)
        else:
            j = random.randint(0, n // 5 - 1)
            if j < K: res[j] = line
rows = []
for line in res:
    d = json.loads(line)
    t = (d.get("reviewText") or "").strip()
    if not t: continue
    rows.append({"asin": d.get("asin"), "rating": int(d.get("overall")), "review_title": (d.get("summary") or "").strip(),
                 "review_text": t, "review_time": d.get("reviewTime"), "verified": d.get("verified")})
with open(OUT, "w", newline="", encoding="utf-8") as g:
    w = csv.DictWriter(g, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("lines scanned:", n, "| sampled rows written:", len(rows))
