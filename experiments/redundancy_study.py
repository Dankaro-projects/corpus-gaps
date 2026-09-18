"""The pre-registered redundancy study (docs/PREREGISTRATION.md).

Confirmatory: four BEIR datasets x two embedding models. Exploratory: the
corpus of the first study, with chunks as the unit.

Run from the repository root:
    uv run --extra embed --extra analysis python experiments/redundancy_study.py
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data-cache"
OUT = ROOT / "results" / "redundancy"
DATASETS = ["scifact", "nfcorpus", "fiqa", "scidocs"]
MODELS = {"bge": "BAAI/bge-small-en-v1.5", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
N_BOOT = 2000
RNG = np.random.default_rng(20260918)


# ------------------------------------------------------------------ data
def load_beir(name):
    CACHE.mkdir(exist_ok=True)
    z = CACHE / f"{name}.zip"
    if not z.exists():
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(BEIR_URL.format(name), z)
    with zipfile.ZipFile(z) as zf:
        def read(member):
            return zf.read(f"{name}/{member}").decode("utf-8")
        corpus = [json.loads(l) for l in read("corpus.jsonl").splitlines() if l.strip()]
        queries = {q["_id"]: q["text"] for q in
                   (json.loads(l) for l in read("queries.jsonl").splitlines() if l.strip())}
        qrels = {}
        for row in csv.DictReader(io.StringIO(read("qrels/test.tsv")), delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    doc_ids = [d["_id"] for d in corpus]
    texts = [(d.get("title") or "") + ". " + (d.get("text") or "") for d in corpus]
    qids = [q for q in qrels if q in queries]
    return doc_ids, texts, [queries[q] for q in qids], [qrels[q] for q in qids]


def load_first_corpus():
    con = sqlite3.connect(ROOT / "corpus.db")
    chunks = con.execute("SELECT c.id, c.doc_id, c.text FROM chunks c ORDER BY c.id").fetchall()
    qs = [json.loads(l) for l in open(ROOT / "experiments" / "questions.jsonl")]
    qs = [q for q in qs if q["target_docs"]]
    doc_of = {cid: did for cid, did, _ in chunks}
    ids = [cid for cid, _, _ in chunks]
    rel = [{cid for cid in ids if doc_of[cid] in set(q["target_docs"])} for q in qs]
    return ids, [t for _, _, t in chunks], [q["question"] for q in qs], rel


# ------------------------------------------------------------ embedding
_models = {}


def encoder(key):
    if key not in _models:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _models[key] = SentenceTransformer(MODELS[key], device=dev)
    return _models[key]


def embed_docs(tag, key, texts):
    path = CACHE / f"{tag}.{key}.docs.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    v = encoder(key).encode(texts, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(path, v)
    print(f"  embedded {len(texts)} documents for {tag} with {key} in {time.time() - t0:.0f} s", flush=True)
    return v


def embed_queries(key, queries):
    prefix = "Represent this sentence for searching relevant passages: " if key == "bge" else ""
    return encoder(key).encode([prefix + q for q in queries], batch_size=64, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


# ------------------------------------------------------------- measures
def per_query(D, Q, doc_ids, rel):
    idx = {d: i for i, d in enumerate(doc_ids)}
    rows = []
    for qi, T in enumerate(rel):
        t_idx = np.array([idx[d] for d in T if d in idx])
        if len(t_idx) == 0:
            continue
        sims = D @ Q[qi]
        mask = np.ones(len(doc_ids), bool)
        mask[t_idx] = False
        s_p, s_a = float(sims.max()), float(sims[mask].max())
        cos_qt = sims[t_idx]
        TD = D[t_idx] @ D.T                      # relevant documents against all documents
        TD[:, ~mask] = -np.inf                  # neighbours must lie outside T
        r_t = TD.max(axis=1)
        valid = (cos_qt > 0) & (r_t > 0)
        bounds = cos_qt * r_t - np.sqrt(np.clip(1 - cos_qt ** 2, 0, None) * np.clip(1 - r_t ** 2, 0, None))
        rows.append({"s_p": s_p, "s_t": float(cos_qt.max()), "s_a": s_a, "delta": s_p - s_a,
                     "r": float(r_t.max()), "b": float(bounds[valid].max()) if valid.any() else None,
                     "n_relevant": int(len(t_idx))})
    return rows


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def analyse(rows):
    s_p = np.array([r["s_p"] for r in rows]); s_a = np.array([r["s_a"] for r in rows])
    s_t = np.array([r["s_t"] for r in rows]); d = np.array([r["delta"] for r in rows])
    r = np.array([r["r"] for r in rows]); n = len(rows)
    res = {"n_queries": n}

    res["detection_auroc"] = round(auroc(s_p, s_a), 3)
    boots = [auroc(s_p[i], s_a[i]) for i in (RNG.integers(0, n, n) for _ in range(N_BOOT))]
    res["detection_auroc_ci"] = ci(boots)

    # H1: Spearman(r, delta) < 0
    rho = stats.spearmanr(r, d).statistic
    boots = [stats.spearmanr(r[i], d[i]).statistic for i in (RNG.integers(0, n, n) for _ in range(N_BOOT))]
    res["H1_spearman_r_delta"] = round(float(rho), 3)
    res["H1_ci"] = ci(boots)
    res["H1_supported"] = bool(res["H1_ci"][1] < 0)

    # H2: AUROC in least redundant third minus most redundant third
    lo_cut, hi_cut = np.quantile(r, [1 / 3, 2 / 3])
    low, high = r <= lo_cut, r > hi_cut
    res["H2_auroc_low_redundancy"] = round(auroc(s_p[low], s_a[low]), 3)
    res["H2_auroc_high_redundancy"] = round(auroc(s_p[high], s_a[high]), 3)
    res["_H2_arrays"] = (s_p[low], s_a[low], s_p[high], s_a[high])

    # H3: R2 gain from adding r to s_t when predicting s_a
    def r2_gain(idx):
        y = s_a[idx]
        X1 = np.column_stack([np.ones(len(idx)), s_t[idx]])
        X2 = np.column_stack([X1, r[idx]])
        def r2(X):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            return 1 - resid.var() / y.var()
        return r2(X2) - r2(X1)
    res["H3_r2_gain"] = round(float(r2_gain(np.arange(n))), 3)
    res["H3_ci"] = ci([r2_gain(RNG.integers(0, n, n)) for _ in range(N_BOOT)])
    res["H3_supported"] = bool(res["H3_ci"][0] > 0)

    # H4: s_a >= b
    b = [(x["s_a"], x["b"]) for x in rows if x["b"] is not None]
    res["H4_checked"] = len(b)
    res["H4_excluded"] = n - len(b)
    res["H4_violations"] = int(sum(sa < bb - 1e-5 for sa, bb in b))
    slack = np.array([sa - bb for sa, bb in b])
    res["bound_slack_median"] = round(float(np.median(slack)), 3) if len(slack) else None
    res["bound_slack_p90"] = round(float(np.percentile(slack, 90)), 3) if len(slack) else None
    res["redundancy_median"] = round(float(np.median(r)), 3)
    res["delta_median"] = round(float(np.median(d)), 4)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"confirmatory": {}, "exploratory": {}, "dropped": []}
    h2_pool = []
    jobs = [(name, True) for name in DATASETS] + [("first_study_corpus", False)]
    for name, confirmatory in jobs:
        try:
            doc_ids, texts, queries, rel = (load_beir(name) if confirmatory else load_first_corpus())
        except Exception as e:  # dropped datasets are reported, never silently skipped
            summary["dropped"].append({"dataset": name, "reason": f"{type(e).__name__}: {e}"})
            continue
        print(f"{name}: {len(doc_ids)} documents, {len(queries)} queries", flush=True)
        for key in MODELS:
            D = embed_docs(name, key, texts)
            Q = embed_queries(key, queries)
            rows = per_query(D, Q, doc_ids, rel)
            with open(OUT / f"{name}.{key}.queries.jsonl", "w") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
            res = analyse(rows)
            arrays = res.pop("_H2_arrays")
            if confirmatory:
                h2_pool.append(arrays)
            (summary["confirmatory"] if confirmatory else summary["exploratory"])[f"{name}/{key}"] = res
            print(f"  {key}: " + json.dumps({k: v for k, v in res.items()}), flush=True)

    # H2 pooled: bootstrap the difference, resampling queries within each combination
    diffs = []
    for _ in range(N_BOOT):
        lo_all, hi_all = [], []
        for pl, al, ph, ah in h2_pool:
            il, ih = RNG.integers(0, len(pl), len(pl)), RNG.integers(0, len(ph), len(ph))
            lo_all.append(auroc(pl[il], al[il])); hi_all.append(auroc(ph[ih], ah[ih]))
        diffs.append(np.mean(lo_all) - np.mean(hi_all))
    point = np.mean([auroc(pl, al) for pl, al, _, _ in h2_pool]) - np.mean([auroc(ph, ah) for _, _, ph, ah in h2_pool])
    summary["H2_pooled"] = {"difference_low_minus_high": round(float(point), 3), "ci": ci(diffs),
                            "supported": bool(ci(diffs)[0] > 0), "combinations": len(h2_pool)}

    conf = summary["confirmatory"]
    k = len(conf)
    allowed = -(-3 * k // 8)   # failure threshold scaled as pre-registered, rounded up
    summary["verdicts"] = {
        "combinations": k,
        "failure_threshold": allowed,
        "H1": {"not_supported_in": sum(not v["H1_supported"] for v in conf.values()),
               "supported": sum(not v["H1_supported"] for v in conf.values()) < allowed},
        "H2": summary["H2_pooled"]["supported"],
        "H3": {"not_supported_in": sum(not v["H3_supported"] for v in conf.values()),
               "supported": sum(not v["H3_supported"] for v in conf.values()) < allowed},
        "H4": {"violations": sum(v["H4_violations"] for v in conf.values()),
               "excluded": sum(v["H4_excluded"] for v in conf.values())},
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary["verdicts"], indent=1))
    print(json.dumps(summary["H2_pooled"]))


if __name__ == "__main__":
    main()
