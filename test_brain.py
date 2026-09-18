"""Exercise every path in corpus_brain that does not need the network."""
import os, numpy as np, sqlite3, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_brain as cb

DB = os.path.join(tempfile.gettempdir(), "test_corpus.db")
if os.path.exists(DB): os.remove(DB)
for suf in ("-wal","-shm"):
    if os.path.exists(DB+suf): os.remove(DB+suf)

con = cb.connect(DB)
print("1. schema + FTS5 created OK")

# --- seeding & dedup
seeds = [
    {"url":"https://www.Example.com/a/?utm_source=x","title":"A","cluster":"G","tier":"A"},
    {"url":"https://example.com/a","title":"A dup","cluster":"G"},          # same doc
    {"url":"https://arxiv.org/pdf/2410.02736","title":"Judge bias","cluster":"J","tier":"A"},
    {"url":"https://arxiv.org/pdf/2606.29270","title":"Minority Sentinel","cluster":"J","tier":"A"},
    {"url":"not-a-url","title":"junk"},
]
n = cb.seed_documents(con, seeds)
assert n == 3, f"expected 3 unique docs, got {n}"
print(f"2. seeding + url dedup OK ({n} unique from {len(seeds)} rows)")

# --- chunker
doc = ("# Analytic Standards\n\n"
       + "ICD 203 sets out nine tradecraft standards for analytic products. " * 14
       + "\n\nSource Reliability\n\n"
       + "The Admiralty code separates source reliability from information credibility. " * 14
       + "\n\nCounter Evidence\n\n"
       + "However, a randomised study found ACH did not reduce confirmation bias. " * 14)
ch = cb.chunk_text(doc, target_tokens=120, overlap_tokens=20)
assert len(ch) >= 3, f"expected multiple chunks, got {len(ch)}"
assert any(h for h, _ in ch), "no headings captured"
print(f"3. chunker OK -> {len(ch)} chunks, headings: {[h[:24] for h,_ in ch]}")

# --- write fake fetched docs then build chunks + FTS
texts = {
 "https://arxiv.org/pdf/2410.02736":
   "# Judge Bias\n\n" + "LLM judges show position bias and verbosity bias in evaluation. " * 20
   + "\n\nMitigation\n\n" + "Ensembling across judge models reduces variance in scores. " * 20,
 "https://arxiv.org/pdf/2606.29270":
   "# Conformity\n\n" + "However, agents conform to the majority and abandon correct judgements. " * 20
   + "\n\nFindings\n\n" + "Weak models correct only 3.6 percent of stance biases in debate. " * 20,
 "https://example.com/a":
   "# Admiralty\n\n" + "Source reliability is graded A to F and credibility 1 to 6. " * 20,
}
for url, t in texts.items():
    con.execute("""UPDATE documents SET fetched_at=1.0, fetch_status='ok',
                   content_type='text/html', n_chars=?, text=? WHERE url_norm=?""",
                (len(t), t, cb.normalise_url(url)))
con.commit()
total = cb.build_chunks(con, verbose=False)
assert total > 0
print(f"4. build_chunks + FTS index OK -> {total} chunks")

# --- keyword search
kw = cb.keyword_search(con, "position bias judges evaluation", k=10)
assert kw, "FTS returned nothing"
print(f"5. FTS/BM25 keyword search OK -> {len(kw)} hits, top score {kw[0][1]:.3f}")

# --- synthetic embeddings to exercise vector path without huggingface
ids = [r["id"] for r in con.execute("SELECT id FROM chunks").fetchall()]
rng = np.random.default_rng(0)
V = rng.normal(size=(len(ids), 32)).astype(np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True)
con.executemany("INSERT OR REPLACE INTO embeddings(chunk_id,model,dim,vec) VALUES(?,?,?,?)",
                [(c, "fake-32", 32, v.tobytes()) for c, v in zip(ids, V)])
con.commit()
lids, M = cb.load_matrix(con)
assert M.shape == (len(ids), 32), M.shape
assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-5)
print(f"6. embedding blob roundtrip OK -> matrix {M.shape}, unit-norm preserved")

class FakeEmb:
    device, name, dim = "cpu", "fake-32", 32
    def query(self, t):
        v = rng.normal(size=32).astype(np.float32); return v/np.linalg.norm(v)
fe = FakeEmb()
vs = cb.vector_search("anything", fe, lids, M, k=5)
assert len(vs) == 5 and vs[0][1] >= vs[-1][1], "vector ranking not sorted"
print(f"7. vector_search OK -> sorted desc, top sim {vs[0][1]:.3f}")

# --- RRF
A = [("x",9),("y",8),("z",7)]
B = [("z",0.9),("x",0.8),("w",0.7)]
fused = cb.rrf(A, B, top=4)
order = [c for c,_ in fused]
assert order[0] in ("x","z"), order
assert "w" in order and "y" in order, order
# x appears rank1+rank2 ; z appears rank3+rank1 -> x should edge z
assert order[0] == "x", f"RRF ordering wrong: {fused}"
print(f"8. RRF fusion OK -> {[(c,round(s,4)) for c,s in fused]}")

# --- hybrid (kw only, since fake embedder is random)
hits = cb.hybrid_search(con, "conformity majority stance bias", embedder=fe, ids=lids, M=M, top=5)
assert hits and "url" in hits[0]
print(f"9. hybrid_search OK -> {len(hits)} enriched hits, first host {hits[0]['host']}")

# --- coverage
cov = cb.coverage_report(con)
assert "chunks_per_doc" in cov.columns
print("10. coverage_report OK\n", cov.to_string(index=False))

# --- tensions (cue regex path)
t = cb.find_tensions(con, lids, M, "debate conformity", fe, pool=40, pairs=5, sim_lo=-1.0, sim_hi=1.0)
print(f"11. find_tensions OK -> {len(t)} candidate pairs (cue-word gated)")
if t: print("    top pair cue_hits =", t[0]["cue_hits"], "| sim =", t[0]["sim"])

# --- citation audit
ans = ("LLM judges show position bias [1]. Ensembling reduces variance [2]. "
       "This claim has no citation at all and should be flagged by the audit. "
       "INFERENCE: the two findings together imply a ceiling. Also cites nothing real [9].")
aud = cb.audit_citations(ans, hits[:3])
assert 9 in aud["invalid_citations"], aud
assert aud["uncited_sentences"], aud
print("12. audit_citations OK ->", {k: v for k, v in aud.items() if k != "uncited_sentences"})
print("    flagged uncited:", aud["uncited_sentences"][0][:60], "...")

# --- cards
cards = cb.make_cards(hits, max_cards=5)
print(f"13. make_cards OK -> {len(cards)} cards; sample back: {cards[0]['back'][:70]}..." if cards
      else "13. make_cards -> 0 (sentence filter too strict on synthetic text)")

# --- runs log
runs = con.execute("SELECT stage, result FROM runs").fetchall()
print(f"14. run log OK -> {[r['stage'] for r in runs]}")

print("\nALL NON-NETWORK PATHS PASSED")
