"""Does a retrieval system know what its corpus does not contain?

For every question the corpus is searched in several conditions, and the best
match score is recorded:

  present         the full corpus, which contains the target document
  doc_absent      the target document (and any copy of it) removed; the rest
                  of the corpus, on related topics, stays
  topic_absent    every document of the target's topic removed
  far             questions on subjects the corpus never covers (full corpus)

Ground truth comes from the construction, not from judgement. A good coverage
signal scores `present` above each of the absent conditions.

Signals compared:
  vector:bge      best cosine similarity, BAAI/bge-small-en-v1.5
  vector:minilm   best cosine similarity, all-MiniLM-L6-v2
  keyword         best BM25 score (SQLite FTS5)
  random          a random number: the floor any signal must beat

Run from the repository root after experiments/build_corpus.py:
    uv run --extra embed --extra analysis python experiments/run_gap_study.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import corpus_brain as cb  # noqa: E402

RNG = np.random.default_rng(20260918)
N_BOOT = 2000
N_SPLITS = 1000
MODELS = {"bge": "BAAI/bge-small-en-v1.5", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
OUT = ROOT / "results"

# A chunk is treated as part of a reference list when it carries at least this
# many citation markers per 1,000 characters. Checked by hand on a sample; see
# the report.
REF_PATTERN = re.compile(
    r"et al\.|arXiv|doi\.org|\bdoi:|Proceedings of|In Proc\.|\(\s?(?:19|20)\d{2}[a-z]?\s?\)"
    r"|\b(?:19|20)\d{2}[a-z]?\.\s|pp\.\s?\d|vol\.\s?\d|https?://", re.I)
REF_DENSITY = 4.0


def ref_density(text: str) -> float:
    return 1000 * len(REF_PATTERN.findall(text)) / max(len(text), 1)


def auroc(pos, neg) -> float:
    """Probability that a random positive outscores a random negative (ties 0.5)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def boot_auroc(pos, neg, paired: bool):
    """AUROC with a 95% bootstrap interval, resampling questions."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    est = auroc(pos, neg)
    vals = []
    for _ in range(N_BOOT):
        if paired:
            i = RNG.integers(0, len(pos), len(pos))
            vals.append(auroc(pos[i], neg[i]))
        else:
            vals.append(auroc(pos[RNG.integers(0, len(pos), len(pos))],
                              neg[RNG.integers(0, len(neg), len(neg))]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(est, 3), round(float(lo), 3), round(float(hi), 3)


def threshold_transfer(pos, neg):
    """Fit the accuracy-maximising threshold on a random half of the question
    pairs, apply it to the other half. Returns mean held-out accuracy and a 95%
    interval over splits, plus the spread of the fitted thresholds."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n = len(pos)
    accs, ths = [], []
    for _ in range(N_SPLITS):
        idx = RNG.permutation(n)
        tr, te = idx[: n // 2], idx[n // 2:]
        cands = np.unique(np.concatenate([pos[tr], neg[tr]]))
        best_t, best_a = cands[0], -1.0
        for t in cands:
            a = ((pos[tr] >= t).sum() + (neg[tr] < t).sum()) / (2 * len(tr))
            if a > best_a:
                best_t, best_a = t, a
        accs.append(((pos[te] >= best_t).sum() + (neg[te] < best_t).sum()) / (2 * len(te)))
        ths.append(best_t)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return {"heldout_accuracy": round(float(np.mean(accs)), 3),
            "ci": [round(float(lo), 3), round(float(hi), 3)],
            "threshold_median": round(float(np.median(ths)), 3),
            "threshold_range": [round(float(np.percentile(ths, 5)), 3),
                                round(float(np.percentile(ths, 95)), 3)]}


def fixed_threshold_accuracy(pos, neg, t):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return round(float(((pos >= t).sum() + (neg < t).sum()) / (len(pos) + len(neg))), 3)


def main():
    con = cb.connect(str(ROOT / "corpus.db"))
    con.row_factory = sqlite3.Row
    questions = [json.loads(l) for l in open(ROOT / "experiments" / "questions.jsonl")]
    doc_cluster = {r["id"]: r["cluster"] for r in
                   con.execute("SELECT id, cluster FROM documents WHERE fetch_status='ok'")}
    chunk_doc = {r["id"]: r["doc_id"] for r in con.execute("SELECT id, doc_id FROM chunks")}
    chunk_text = {r["id"]: r["text"] for r in con.execute("SELECT id, text FROM chunks")}
    is_ref = {c: ref_density(t) >= REF_DENSITY for c, t in chunk_text.items()}

    embedders, mats = {}, {}
    for key, name in MODELS.items():
        ids, M = cb.load_matrix(con, model=name)
        embedders[key] = cb.Embedder(name, device="cpu")
        mats[key] = (np.array(ids), M)

    def excluded(q, condition):
        if condition in ("present", "far"):
            return set()
        if condition == "doc_absent":
            return set(q["target_docs"])
        topic = doc_cluster[q["target_docs"][0]]
        return {d for d, c in doc_cluster.items() if c == topic}

    def vector_best(key, qvec, drop_docs, drop_refs):
        ids, M = mats[key]
        keep = np.array([(chunk_doc[c] not in drop_docs) and not (drop_refs and is_ref[c])
                         for c in ids])
        sims = M[keep] @ qvec
        return float(sims.max()), ids[keep][np.argsort(-sims)[:50]].tolist()

    def keyword_ranked(question, drop_docs, drop_refs):
        hits = cb.keyword_search(con, question, k=3000)
        return [(c, s) for c, s in hits
                if chunk_doc.get(c) not in drop_docs and not (drop_refs and is_ref.get(c))]

    rows = []
    qvecs = {k: {q["id"]: embedders[k].query(q["question"]) for q in questions} for k in MODELS}
    for drop_refs in (False, True):
        for q in questions:
            conds = ["far"] if not q["target_docs"] else ["present", "doc_absent", "topic_absent"]
            for cond in conds:
                drop = excluded(q, cond)
                rec = {"question": q["id"], "condition": cond, "refs_filtered": drop_refs,
                       "topic": doc_cluster[q["target_docs"][0]] if q["target_docs"] else None}
                for key in MODELS:
                    best, top = vector_best(key, qvecs[key][q["id"]], drop, drop_refs)
                    rec[f"vector:{key}"] = round(best, 4)
                    if cond == "present":
                        rec[f"hit5:vector:{key}"] = any(chunk_doc[c] in q["target_docs"] for c in top[:5])
                        rec[f"_top:{key}"] = top
                kw = keyword_ranked(q["question"], drop, drop_refs)
                rec["keyword"] = round(kw[0][1], 4) if kw else 0.0
                rec["random"] = float(RNG.random())
                if cond == "present":
                    rec["hit5:keyword"] = any(chunk_doc[c] in q["target_docs"] for c, _ in kw[:5])
                    fused = cb.rrf(kw[:50], [(c, 0) for c in rec["_top:bge"]], top=5)
                    rec["hit5:hybrid"] = any(chunk_doc[c] in q["target_docs"] for c, _ in fused)
                rows.append({k: v for k, v in rec.items() if not k.startswith("_top")})

    OUT.mkdir(exist_ok=True)
    with open(OUT / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    signals = ["vector:bge", "vector:minilm", "keyword", "random"]
    summary = {"corpus": {
        "documents_seeded": con.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "documents_ok": len(doc_cluster),
        "fetch_status": dict(con.execute("SELECT fetch_status, COUNT(*) FROM documents "
                                         "GROUP BY fetch_status").fetchall()),
        "chunks": len(chunk_doc),
        "reference_chunks": int(sum(is_ref.values())),
        "questions_on_documents": sum(1 for q in questions if q["target_docs"]),
        "questions_far": sum(1 for q in questions if not q["target_docs"]),
    }, "detection": {}, "retrieval_hit_at_5": {}, "threshold": {}}

    for drop_refs in (False, True):
        tag = "refs_filtered" if drop_refs else "all_chunks"
        R = [r for r in rows if r["refs_filtered"] == drop_refs]
        by = lambda cond: sorted((r for r in R if r["condition"] == cond), key=lambda r: r["question"])
        pres, dabs, tabs, far = by("present"), by("doc_absent"), by("topic_absent"), by("far")
        det = {}
        for s in signals:
            P = [r[s] for r in pres]
            det[s] = {
                "present_vs_doc_absent": boot_auroc(P, [r[s] for r in dabs], paired=True),
                "present_vs_topic_absent": boot_auroc(P, [r[s] for r in tabs], paired=True),
                "present_vs_far": boot_auroc(P, [r[s] for r in far], paired=False),
                "score_ranges": {c: [round(min(r[s] for r in L), 3), round(max(r[s] for r in L), 3)]
                                 for c, L in (("present", pres), ("doc_absent", dabs),
                                              ("topic_absent", tabs), ("far", far))},
            }
        summary["detection"][tag] = det
        summary["retrieval_hit_at_5"][tag] = {
            k: f"{sum(r[k] for r in pres)}/{len(pres)}"
            for k in ("hit5:vector:bge", "hit5:vector:minilm", "hit5:keyword", "hit5:hybrid")}
        th = {}
        for s in ("vector:bge", "vector:minilm", "keyword"):
            P = [r[s] for r in pres]
            th[s] = {"doc_absent": threshold_transfer(P, [r[s] for r in dabs]),
                     "topic_absent": threshold_transfer(P, [r[s] for r in tabs])}
        for s in ("vector:bge", "vector:minilm"):
            P = [r[s] for r in pres]
            th[s]["fixed_0.72"] = {
                "doc_absent": fixed_threshold_accuracy(P, [r[s] for r in dabs], 0.72),
                "topic_absent": fixed_threshold_accuracy(P, [r[s] for r in tabs], 0.72),
                "far": fixed_threshold_accuracy(P, [r[s] for r in far], 0.72)}
        summary["threshold"][tag] = th

    # Topic structure: do unsupervised clusters recover the six topics, with and
    # without the reference chunks?
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score
    ids, M = mats["bge"]
    topics = np.array([doc_cluster[chunk_doc[c]] for c in ids])
    refs = np.array([is_ref[c] for c in ids])
    clus = {}
    for tag, mask in (("all_chunks", np.ones(len(ids), bool)), ("refs_filtered", ~refs)):
        amis, ref_dominated = [], []
        for seed in range(5):
            lab = KMeans(n_clusters=14, n_init=10, random_state=seed).fit_predict(M[mask])
            amis.append(adjusted_mutual_info_score(topics[mask], lab))
            if tag == "all_chunks":
                ref_dominated.append(sum(refs[mask][lab == k].mean() > 0.5 for k in range(14)))
        clus[tag] = {"ami_vs_topics_mean": round(float(np.mean(amis)), 3),
                     "ami_range": [round(float(min(amis)), 3), round(float(max(amis)), 3)],
                     "chunks": int(mask.sum())}
        if ref_dominated:
            clus[tag]["clusters_mostly_references_of_14"] = [int(x) for x in ref_dominated]
    summary["clustering"] = clus

    # A sample of flagged and unflagged chunks, for checking the reference rule by hand.
    flagged = [c for c in ids if is_ref[c]]
    unflagged = [c for c in ids if not is_ref[c]]
    sample = {"flagged": [chunk_text[c][:300] for c in RNG.choice(flagged, 15, replace=False)],
              "unflagged_high_density": [chunk_text[c][:300] for c in sorted(
                  unflagged, key=lambda c: -ref_density(chunk_text[c]))[:10]]}
    json.dump(sample, open(OUT / "reference_rule_sample.json", "w"), indent=1)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
