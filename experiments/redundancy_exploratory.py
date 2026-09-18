"""Exploratory follow up to the pre-registered study. Not pre-registered.

Question: is detection near chance because, for most queries, the best match
is not a relevant document at all? If so, removing the relevant documents
cannot change the best score, and no signal based on it can detect the gap.
Also: does H1 hold among queries whose best match is relevant?
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "results" / "redundancy"
RNG = np.random.default_rng(7)
res = {}
for f in sorted(OUT.glob("*.queries.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    d = np.array([r["delta"] for r in rows]); r = np.array([r["r"] for r in rows])
    top_rel = d > 1e-9
    sub = {"queries": len(rows), "best_match_is_relevant": int(top_rel.sum()),
           "share_best_match_relevant": round(float(top_rel.mean()), 3)}
    if top_rel.sum() >= 20:
        rr, dd = r[top_rel], d[top_rel]; n = len(rr)
        rho = stats.spearmanr(rr, dd).statistic
        boots = [stats.spearmanr(rr[i], dd[i]).statistic for i in (RNG.integers(0, n, n) for _ in range(2000))]
        sub["H1_on_those_queries"] = round(float(rho), 3)
        sub["ci"] = [round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])]
    res[f.name.replace(".queries.jsonl", "")] = sub
json.dump(res, open(OUT / "exploratory.json", "w"), indent=1)
for k, v in res.items():
    print(k, v)
