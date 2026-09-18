"""The confirmatory test in docs/PREREGISTRATION_2.md.

Run from the repository root:
    uv run --extra embed --extra analysis --with scipy python experiments/confirmatory_study.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_study as rs  # noqa: E402  same embedding, measures and first-plan analysis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "confirmatory"
RNG = np.random.default_rng(20260919)
N_BOOT = 2000
MIN_ANSWERED = 20
# (label, zip name, folder inside the zip)
DATASETS = [("trec-covid", "trec-covid", "trec-covid"),
            ("cqa-android", "cqadupstack", "cqadupstack/android"),
            ("cqa-english", "cqadupstack", "cqadupstack/english"),
            ("cqa-gaming", "cqadupstack", "cqadupstack/gaming"),
            ("cqa-physics", "cqadupstack", "cqadupstack/physics")]


def load(zip_name, folder):
    rs.CACHE.mkdir(exist_ok=True)
    z = rs.CACHE / f"{zip_name}.zip"
    if not z.exists():
        print(f"downloading {zip_name}", flush=True)
        urllib.request.urlretrieve(rs.BEIR_URL.format(zip_name), z)
    with zipfile.ZipFile(z) as zf:
        read = lambda m: zf.read(f"{folder}/{m}").decode("utf-8")
        corpus = [json.loads(l) for l in read("corpus.jsonl").splitlines() if l.strip()]
        queries = {q["_id"]: q["text"] for q in
                   (json.loads(l) for l in read("queries.jsonl").splitlines() if l.strip())}
        qrels = {}
        for row in csv.DictReader(io.StringIO(read("qrels/test.tsv")), delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    qids = [q for q in qrels if q in queries]
    return ([d["_id"] for d in corpus],
            [(d.get("title") or "") + ". " + (d.get("text") or "") for d in corpus],
            [queries[q] for q in qids], [qrels[q] for q in qids])


def ci(v):
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"combinations": {}, "dropped": []}
    h5, h6 = [], []
    for label, zname, folder in DATASETS:
        try:
            doc_ids, texts, queries, rel = load(zname, folder)
        except Exception as e:
            out["dropped"].append({"dataset": label, "reason": f"{type(e).__name__}: {e}"})
            continue
        print(f"{label}: {len(doc_ids)} documents, {len(queries)} queries", flush=True)
        for key in rs.MODELS:
            D = rs.embed_docs(label, key, texts)
            Q = rs.embed_queries(key, queries)
            rows = rs.per_query(D, Q, doc_ids, rel)
            with open(OUT / f"{label}.{key}.queries.jsonl", "w") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
            first_plan = rs.analyse(rows)
            first_plan.pop("_H2_arrays")
            s_p = np.array([x["s_p"] for x in rows]); s_a = np.array([x["s_a"] for x in rows])
            d = np.array([x["delta"] for x in rows]); r = np.array([x["r"] for x in rows])
            ans = d > 1e-9
            res = {"queries": len(rows), "answered": int(ans.sum()),
                   "share_answered": round(float(ans.mean()), 3),
                   "auroc_all": first_plan["detection_auroc"],
                   "auroc_answered": round(rs.auroc(s_p[ans], s_a[ans]), 3) if ans.any() else None,
                   "auroc_not_answered": round(rs.auroc(s_p[~ans], s_a[~ans]), 3) if (~ans).any() else None,
                   "counted": bool(ans.sum() >= MIN_ANSWERED), "first_plan": first_plan}
            if res["counted"]:
                ra, da, pa, aa = r[ans], d[ans], s_p[ans], s_a[ans]
                n = len(ra)
                rho = stats.spearmanr(ra, da).statistic
                boots = [stats.spearmanr(ra[i], da[i]).statistic for i in (RNG.integers(0, n, n) for _ in range(N_BOOT))]
                res["H5_spearman"] = round(float(rho), 3)
                res["H5_ci"] = ci(boots)
                res["H5_supported"] = bool(res["H5_ci"][1] < 0)
                h5.append(float(rho))
                lo_cut, hi_cut = np.quantile(ra, [1 / 3, 2 / 3])
                lo, hi = ra <= lo_cut, ra > hi_cut
                res["H6_auroc_low"] = round(rs.auroc(pa[lo], aa[lo]), 3)
                res["H6_auroc_high"] = round(rs.auroc(pa[hi], aa[hi]), 3)
                h6.append((pa[lo], aa[lo], pa[hi], aa[hi]))
            out["combinations"][f"{label}/{key}"] = res
            print(f"  {key}: " + json.dumps({k: v for k, v in res.items() if k != "first_plan"}), flush=True)

    counted = [v for v in out["combinations"].values() if v["counted"]]
    k = len(counted)
    threshold = -(-3 * k // 8)
    diffs = []
    for _ in range(N_BOOT):
        lo_v, hi_v = [], []
        for pl, al, ph, ah in h6:
            il, ih = RNG.integers(0, len(pl), len(pl)), RNG.integers(0, len(ph), len(ph))
            lo_v.append(rs.auroc(pl[il], al[il])); hi_v.append(rs.auroc(ph[ih], ah[ih]))
        diffs.append(np.mean(lo_v) - np.mean(hi_v))
    point = (np.mean([rs.auroc(pl, al) for pl, al, _, _ in h6])
             - np.mean([rs.auroc(ph, ah) for _, _, ph, ah in h6]))
    not_h5 = sum(not v["H5_supported"] for v in counted)
    out["verdicts"] = {
        "counted_combinations": k, "failure_threshold": threshold,
        "H5": {"not_supported_in": not_h5, "supported": not_h5 < threshold},
        "H6": {"difference_low_minus_high": round(float(point), 3), "ci": ci(diffs),
               "supported": bool(ci(diffs)[0] > 0)},
        "H7": {"mean_spearman": round(float(np.mean(h5)), 3), "supported": bool(np.mean(h5) <= -0.15)},
    }
    json.dump(out, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(out["verdicts"], indent=1))


if __name__ == "__main__":
    main()
