"""
corpus_brain.py — build a searchable, embedded, self-auditing research corpus.

DESIGN CONTRACT
---------------
Everything that enters the database keeps its provenance. Every chunk knows its
document; every document knows the query that found it and when it was fetched.
Nothing is summarised into the store — summaries are derived views, never the
record of truth. That is the whole point: you can always go back to the text.

STAGES
    seed / harvest  -> documents (url, host, title, snippet, query, cluster)
    fetch           -> raw text + extraction metadata
    chunk           -> overlapping windows with heading context
    embed           -> float32 vectors
    index           -> FTS5 keyword index
    search          -> hybrid BM25 + cosine, fused with Reciprocal Rank Fusion
    analyse         -> clusters, contradictions, coverage gaps, claim extraction

WHAT IS AND IS NOT TESTED
    test_brain.py covers schema creation, FTS5, the chunker, RRF fusion, cosine
    search against synthetic vectors and URL normalisation, with no network.
    The study in experiments/ has run seeding, fetching, chunking, embedding
    with two models, keyword and vector search, and clustering on a public
    corpus of 60 URLs. Two stages have never been executed: `harvest` (it has
    never queried a search engine) and grounded answering with Claude (no API
    key was available), so treat their first run as their test.
"""

from __future__ import annotations
import hashlib, json, os, re, sqlite3, time, random, math
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Iterable, Sequence

import numpy as np

TRACKING = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content",
            "gclid","fbclid","mc_cid","mc_eid","ref","source","spm","igshid"}

DEFAULT_DB = "corpus.db"


# ─────────────────────────────────────────────────────────────── url handling
def normalise_url(url: str) -> str:
    """Canonical form for dedup. Differs from the raw URL only in ways that do
    not change what document you get."""
    try:
        p = urlparse(url.strip())
    except Exception:
        return url.strip().lower()
    q = [(k, v) for k, v in parse_qsl(p.query) if k.lower() not in TRACKING]
    return urlunparse((p.scheme.lower(),
                       p.netloc.lower().removeprefix("www."),
                       p.path.rstrip("/") or "/",
                       "", urlencode(sorted(q)), ""))


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return "unknown"


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────── schema
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS documents (
  id            TEXT PRIMARY KEY,          -- sha of normalised url
  url           TEXT NOT NULL,
  url_norm      TEXT NOT NULL UNIQUE,
  host          TEXT NOT NULL,
  title         TEXT,
  snippet       TEXT,
  cluster       TEXT,                      -- semantic cluster label
  found_by      TEXT,                      -- the query that surfaced it
  tier          TEXT,                      -- A/B/C curation tier, optional
  fetched_at    REAL,                      -- epoch, NULL = not fetched
  fetch_status  TEXT,                      -- ok | http_4xx | blocked | empty | error:<type>
  content_type  TEXT,
  n_chars       INTEGER,
  text          TEXT,
  notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_doc_host    ON documents(host);
CREATE INDEX IF NOT EXISTS idx_doc_cluster ON documents(cluster);
CREATE INDEX IF NOT EXISTS idx_doc_status  ON documents(fetch_status);

CREATE TABLE IF NOT EXISTS chunks (
  id        TEXT PRIMARY KEY,              -- sha(doc_id|ordinal)
  doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal   INTEGER NOT NULL,
  heading   TEXT,
  text      TEXT NOT NULL,
  n_tokens  INTEGER,
  UNIQUE(doc_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  model    TEXT NOT NULL,
  dim      INTEGER NOT NULL,
  vec      BLOB NOT NULL,                  -- float32, L2-normalised
  PRIMARY KEY (chunk_id, model)            -- one vector per chunk per model
);

CREATE TABLE IF NOT EXISTS runs (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  stage     TEXT, started REAL, finished REAL, params TEXT, result TEXT
);

CREATE TABLE IF NOT EXISTS notes (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  kind      TEXT,                          -- claim | contradiction | gap | synthesis
  body      TEXT,
  evidence  TEXT,                          -- json list of chunk ids
  created   REAL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  text, heading, chunk_id UNINDEXED, tokenize='porter unicode61'
);
"""


def connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def log_run(con, stage, started, params, result):
    con.execute("INSERT INTO runs(stage,started,finished,params,result) VALUES(?,?,?,?,?)",
                (stage, started, time.time(), json.dumps(params, default=str),
                 json.dumps(result, default=str)))
    con.commit()


# ────────────────────────────────────────────────────────────────── seeding
def seed_documents(con, rows: Iterable[dict]) -> int:
    """rows need at least 'url'. Optional: title, snippet, cluster, found_by, tier."""
    for r in rows:
        u = (r.get("url") or "").strip()
        if not u.startswith("http"):
            continue
        un = normalise_url(u)
        try:
            con.execute(
                """INSERT OR IGNORE INTO documents
                   (id,url,url_norm,host,title,snippet,cluster,found_by,tier)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (sha(un), u, un, host_of(u), r.get("title"), r.get("snippet"),
                 r.get("cluster"), r.get("found_by"), r.get("tier")))
        except sqlite3.IntegrityError:
            pass
    con.commit()
    return con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


# ────────────────────────────────────────────────────────────────── harvest
def harvest(con, queries: Sequence[str], per_query=20, max_per_host=8,
            sleep=(3.0, 7.0), engine="ddg", searx_url=None, target=None,
            cluster_of=None, verbose=True) -> dict:
    """Search, dedup, and insert as unfetched documents.

    NOTE ON RATE LIMITING: DuckDuckGo soft-bans aggressive clients by returning
    an empty result set rather than an error. If you start seeing '+0' on every
    query, you are banned — stop, wait an hour, and raise the sleep window.
    """
    t0 = time.time()
    if engine == "ddg":
        from ddgs import DDGS
        def run(q, n):
            with DDGS() as d:
                return [{"title": r.get("title",""), "url": r.get("href") or r.get("url",""),
                         "snippet": r.get("body","")} for r in d.text(q, max_results=n)]
    else:
        import urllib.request
        def run(q, n):
            url = f"{searx_url.rstrip('/')}/search?" + urlencode(
                {"q": q, "format": "json", "language": "en"})
            req = urllib.request.Request(url, headers={"User-Agent": "corpus-brain/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return [{"title": r.get("title",""), "url": r.get("url",""),
                     "snippet": r.get("content","")} for r in data.get("results", [])[:n]]

    host_count = dict(con.execute("SELECT host, COUNT(*) FROM documents GROUP BY host").fetchall())
    added_total, empty_streak = 0, 0

    for i, q in enumerate(queries, 1):
        if target and con.execute("SELECT COUNT(*) FROM documents").fetchone()[0] >= target:
            if verbose: print("target reached")
            break
        try:
            hits = run(q, per_query)
        except Exception as e:
            if verbose: print(f"[{i}/{len(queries)}] FAIL {type(e).__name__}: {str(e)[:100]}")
            time.sleep(sleep[1] * 3)
            continue

        if not hits:
            empty_streak += 1
            if empty_streak >= 5:
                print("!! five consecutive empty responses — you are probably rate-limited. Stopping.")
                break
        else:
            empty_streak = 0

        added = 0
        for h in hits:
            u = h["url"]
            if not u:
                continue
            host = host_of(u)
            if host_count.get(host, 0) >= max_per_host:
                continue
            un = normalise_url(u)
            cur = con.execute(
                """INSERT OR IGNORE INTO documents
                   (id,url,url_norm,host,title,snippet,cluster,found_by)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (sha(un), u, un, host, h["title"][:300], h["snippet"][:800],
                 (cluster_of or {}).get(q), q))
            if cur.rowcount:
                host_count[host] = host_count.get(host, 0) + 1
                added += 1
        con.commit()
        added_total += added
        if verbose:
            total = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            print(f"[{i}/{len(queries)}] +{added:2d}  total={total:4d}  :: {q[:58]}")
        time.sleep(random.uniform(*sleep))

    res = {"added": added_total,
           "total": con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]}
    log_run(con, "harvest", t0, {"n_queries": len(queries), "engine": engine}, res)
    return res


# ──────────────────────────────────────────────────────────────────── fetch
def fetch_all(con, limit=None, timeout=25, workers=8, sleep_per_host=1.5,
              user_agent="corpus-gaps/0.1 (research; contact: you@example.com)",
              verbose=True) -> dict:
    """Fetch page text. trafilatura for HTML, pypdf for PDF.

    Politeness: one in-flight request per host and a per-host delay. Set a real
    contact address in the user agent — it is what stops you being blocked.
    """
    import concurrent.futures as cf, urllib.request, urllib.error, io
    from collections import defaultdict
    import threading

    t0 = time.time()
    rows = con.execute(
        "SELECT id,url FROM documents WHERE fetched_at IS NULL"
        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    if verbose: print(f"fetching {len(rows)} documents")

    host_locks = defaultdict(threading.Lock)
    try:
        import trafilatura
    except ImportError:
        trafilatura = None

    def _one(row):
        url = row["url"]
        h = host_of(url)
        ctype = ""
        with host_locks[h]:
            time.sleep(sleep_per_host)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": user_agent})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    raw = resp.read(8_000_000)
            except urllib.error.HTTPError as e:
                return row["id"], f"http_{e.code}", "", ""
            except Exception as e:
                return row["id"], f"error:{type(e).__name__}", "", ""

        # Extraction runs OUTSIDE the host lock, and must never raise: one
        # malformed page would otherwise kill the whole ThreadPoolExecutor.map
        # and take every not-yet-written result with it.
        if "pdf" in ctype.lower() or url.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                r = PdfReader(io.BytesIO(raw))
                txt = "\n\n".join((p.extract_text() or "") for p in r.pages)
            except Exception as e:
                return row["id"], f"error:pdf_{type(e).__name__}", ctype, ""
        else:
            try:
                html = raw.decode("utf-8", "ignore")
                txt = ""
                if trafilatura:
                    txt = trafilatura.extract(html, include_comments=False,
                                              include_tables=True, favor_recall=True) or ""
                if not txt:  # crude fallback
                    txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html,
                                 flags=re.S | re.I)
                    txt = re.sub(r"<[^>]+>", " ", txt)
                    txt = re.sub(r"\s+", " ", txt)
            except Exception as e:
                return row["id"], f"error:extract_{type(e).__name__}", ctype, ""
        txt = (txt or "").strip()
        return row["id"], ("ok" if len(txt) > 400 else "empty"), ctype, txt

    def one(row):
        try:
            return _one(row)
        except Exception as e:                      # last-resort net
            return row["id"], f"error:{type(e).__name__}", "", ""

    ok = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for n, (doc_id, status, ctype, txt) in enumerate(ex.map(one, rows), 1):
            con.execute("""UPDATE documents SET fetched_at=?, fetch_status=?,
                           content_type=?, n_chars=?, text=? WHERE id=?""",
                        (time.time(), status, ctype, len(txt), txt, doc_id))
            if status == "ok": ok += 1
            if n % 25 == 0:
                con.commit()
                if verbose: print(f"  {n}/{len(rows)}  ok={ok}")
    con.commit()

    res = dict(con.execute(
        "SELECT fetch_status, COUNT(*) FROM documents WHERE fetched_at IS NOT NULL "
        "GROUP BY fetch_status").fetchall())
    log_run(con, "fetch", t0, {"n": len(rows)}, res)
    if verbose: print("fetch status:", res)
    return res


# ──────────────────────────────────────────────────────────────────── chunk
HEADING_RE = re.compile(r"^\s{0,3}(#{1,4}\s+.+|[A-Z][A-Za-z0-9 ,'\-()/]{6,80})\s*$")

def _split_long(p: str, limit: int) -> list[str]:
    """Break an over-long paragraph on sentence boundaries, hard-slicing only
    when a single sentence exceeds the limit.

    Without this, a page whose extractor emits the whole body as one paragraph
    becomes one enormous chunk — and the embedder silently truncates it at its
    own token limit, so everything past the first few hundred tokens is stored
    but never searchable.
    """
    if len(p) <= limit:
        return [p]
    out, buf = [], ""
    for s in re.split(r"(?<=[.!?])\s+", p):
        while len(s) > limit:
            out.append(s[:limit]); s = s[limit:]
        if not s:
            continue
        if buf and len(buf) + len(s) + 1 > limit:
            out.append(buf); buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, target_tokens=320, overlap_tokens=60) -> list[tuple[str, str]]:
    """Return [(heading, chunk_text)]. Tokens approximated at 4 chars each —
    close enough for chunking, not close enough for billing."""
    tgt, ovl = target_tokens * 4, overlap_tokens * 4
    paras = [q for p in re.split(r"\n\s*\n", text) if p.strip()
             for q in _split_long(p.strip(), tgt)]
    out, buf, heading, cur_head = [], "", "", ""
    for p in paras:
        if len(p) < 90 and HEADING_RE.match(p):
            cur_head = p.lstrip("# ").strip()
        if len(buf) + len(p) + 2 > tgt and buf:
            out.append((heading or cur_head, buf.strip()))
            buf = buf[-ovl:] if ovl else ""
            heading = cur_head
        if not buf:
            heading = heading or cur_head
        buf += ("\n\n" if buf else "") + p
    if buf.strip():
        out.append((heading or cur_head, buf.strip()))
    return [(h, t) for h, t in out if len(t) > 150]


def build_chunks(con, verbose=True) -> int:
    t0 = time.time()
    rows = con.execute("""SELECT d.id, d.text FROM documents d
                          WHERE d.fetch_status='ok'
                            AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id=d.id)"""
                       ).fetchall()
    n = 0
    for row in rows:
        for i, (head, txt) in enumerate(chunk_text(row["text"])):
            cid = sha(f"{row['id']}|{i}")
            con.execute("""INSERT OR IGNORE INTO chunks(id,doc_id,ordinal,heading,text,n_tokens)
                           VALUES(?,?,?,?,?,?)""",
                        (cid, row["id"], i, head, txt, len(txt)//4))
            con.execute("INSERT INTO chunk_fts(text,heading,chunk_id) VALUES(?,?,?)",
                        (txt, head or "", cid))
            n += 1
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    log_run(con, "chunk", t0, {"docs": len(rows)}, {"new": n, "total": total})
    if verbose: print(f"chunked {len(rows)} docs -> +{n} chunks (total {total})")
    return total


# ──────────────────────────────────────────────────────────────────── embed
class Embedder:
    """Local sentence-transformers by default. Colab's free GPU handles a
    100k-chunk corpus in minutes; CPU will take roughly an hour."""
    def __init__(self, model="BAAI/bge-small-en-v1.5", device=None, batch=64):
        from sentence_transformers import SentenceTransformer
        import torch
        self.name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = SentenceTransformer(model, device=self.device)
        self.dim = self.m.get_sentence_embedding_dimension()
        self.batch = batch
        # bge models want an instruction prefix on the QUERY side only
        self.query_prefix = ("Represent this sentence for searching relevant passages: "
                             if "bge" in model.lower() else "")

    def docs(self, texts):
        v = self.m.encode(list(texts), batch_size=self.batch, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True)
        return v.astype(np.float32)

    def query(self, text):
        v = self.m.encode([self.query_prefix + text], convert_to_numpy=True,
                          normalize_embeddings=True)
        return v.astype(np.float32)[0]


def embed_chunks(con, embedder: "Embedder", batch=512, verbose=True) -> int:
    t0 = time.time()
    rows = con.execute("""SELECT c.id, c.text FROM chunks c
                          LEFT JOIN embeddings e ON e.chunk_id=c.id AND e.model=?
                          WHERE e.chunk_id IS NULL""", (embedder.name,)).fetchall()
    if verbose: print(f"embedding {len(rows)} chunks on {embedder.device} with {embedder.name}")
    for i in range(0, len(rows), batch):
        part = rows[i:i+batch]
        vecs = embedder.docs([r["text"] for r in part])
        con.executemany("INSERT OR REPLACE INTO embeddings(chunk_id,model,dim,vec) VALUES(?,?,?,?)",
                        [(r["id"], embedder.name, embedder.dim, v.tobytes())
                         for r, v in zip(part, vecs)])
        con.commit()
        if verbose: print(f"  {min(i+batch,len(rows))}/{len(rows)}")
    total = con.execute("SELECT COUNT(*) FROM embeddings WHERE model=?",
                        (embedder.name,)).fetchone()[0]
    log_run(con, "embed", t0, {"model": embedder.name}, {"total": total})
    return total


def load_matrix(con, model=None):
    """All vectors as one (n, dim) float32 array plus the chunk id list.
    Brute force is fine to roughly 200k chunks; beyond that use faiss.

    If the table holds more than one model, the largest set wins — mixing
    dimensions in one matrix is not a thing, and a silent reshape error at
    search time is worse than a stated choice.
    """
    if model is None:
        counts = con.execute(
            "SELECT model, COUNT(*) n FROM embeddings GROUP BY model ORDER BY n DESC"
        ).fetchall()
        if not counts:
            return [], np.zeros((0, 0), dtype=np.float32)
        model = counts[0]["model"]
        if len(counts) > 1:
            print(f"load_matrix: {len(counts)} models in embeddings; using "
                  f"'{model}' ({counts[0]['n']} vectors). Others ignored: "
                  + ", ".join(f"{r['model']}({r['n']})" for r in counts[1:]))
    rows = con.execute(
        "SELECT chunk_id, dim, vec FROM embeddings WHERE model = :m", {"m": model}
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dim = rows[0]["dim"]
    ids = [r["chunk_id"] for r in rows]
    M = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32).reshape(len(rows), dim)
    return ids, M


# ─────────────────────────────────────────────────────────────────── search
def _fts_escape(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9']+", q)
    return " OR ".join(f'"{t}"' for t in toks if len(t) > 2) or '"' + q[:40] + '"'


def keyword_search(con, query, k=50):
    rows = con.execute(
        """SELECT chunk_id, bm25(chunk_fts) AS score FROM chunk_fts
           WHERE chunk_fts MATCH ? ORDER BY score LIMIT ?""",
        (_fts_escape(query), k)).fetchall()
    return [(r["chunk_id"], -r["score"]) for r in rows]   # bm25: lower is better


def vector_search(query, embedder, ids, M, k=50):
    if len(ids) == 0 or getattr(M, "ndim", 0) != 2 or M.shape[0] == 0:
        return []
    qv = embedder.query(query)
    if M.shape[1] != qv.shape[0]:
        raise ValueError(f"embedder dim {qv.shape[0]} != matrix dim {M.shape[1]} "
                         "— re-run load_matrix after changing embedding model")
    sims = M @ qv
    top = np.argpartition(-sims, min(k, len(sims)-1))[:k]
    top = top[np.argsort(-sims[top])]
    return [(ids[i], float(sims[i])) for i in top]


def rrf(*rankings, k=60, top=20):
    """Reciprocal Rank Fusion. Combines rankings without needing their scores
    to be on comparable scales, which BM25 and cosine emphatically are not."""
    scores = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]


def hybrid_search(con, query, embedder=None, ids=None, M=None, k=40, top=12):
    kw = keyword_search(con, query, k)
    vec = vector_search(query, embedder, ids, M, k) if embedder is not None and len(ids or []) else []
    fused = rrf(kw, vec, top=top) if vec else [(c, s) for c, s in kw[:top]]
    out = []
    for cid, score in fused:
        r = con.execute("""SELECT c.text, c.heading, d.title, d.url, d.host, d.cluster
                           FROM chunks c JOIN documents d ON d.id=c.doc_id
                           WHERE c.id=?""", (cid,)).fetchone()
        if r:
            out.append({"chunk_id": cid, "score": round(score, 5), "text": r["text"],
                        "heading": r["heading"], "title": r["title"], "url": r["url"],
                        "host": r["host"], "cluster": r["cluster"]})
    return out


# ───────────────────────────────────────────────────────────────── analysis
def coverage_report(con):
    import pandas as pd
    q = """SELECT COALESCE(cluster,'(unassigned)') AS cluster,
                  COUNT(*) AS docs,
                  SUM(fetch_status='ok') AS fetched_ok,
                  SUM(fetch_status IS NOT NULL AND fetch_status<>'ok') AS fetch_failed,
                  COUNT(DISTINCT host) AS hosts
           FROM documents GROUP BY 1 ORDER BY docs DESC"""
    df = pd.read_sql_query(q, con)
    ch = pd.read_sql_query(
        """SELECT COALESCE(d.cluster,'(unassigned)') AS cluster, COUNT(c.id) AS chunks
           FROM chunks c JOIN documents d ON d.id=c.doc_id GROUP BY 1""", con)
    df = df.merge(ch, on="cluster", how="left").fillna({"chunks": 0})
    df["chunks_per_doc"] = (df["chunks"] / df["docs"].clip(lower=1)).round(1)
    df["host_concentration"] = (df["docs"] / df["hosts"].clip(lower=1)).round(2)
    return df


def cluster_corpus(con, ids, M, n_clusters=14, sample_terms=8):
    """KMeans over chunk vectors, labelled by distinctive TF-IDF terms.
    This is topic DISCOVERY — it tells you what the corpus actually contains,
    which is usually not what you thought you were searching for."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(ids) < n_clusters * 3:
        raise ValueError(f"need at least {n_clusters*3} chunks, have {len(ids)}")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(M)
    texts = {r["id"]: r["text"] for r in
             con.execute("SELECT id, text FROM chunks").fetchall()}
    docs = [texts.get(i, "") for i in ids]
    tf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
    X = tf.fit_transform(docs)
    vocab = np.array(tf.get_feature_names_out())
    out = []
    for c in range(n_clusters):
        mask = km.labels_ == c
        if mask.sum() == 0:
            continue
        centroid = np.asarray(X[mask].mean(axis=0)).ravel()
        terms = vocab[np.argsort(-centroid)[:sample_terms]]
        out.append({"cluster": c, "n_chunks": int(mask.sum()),
                    "terms": ", ".join(terms),
                    "example_chunk": ids[int(np.where(mask)[0][0])]})
    return out, km.labels_


CONTRA_CUES = re.compile(
    r"\b(however|but |contrary|contradict|disagree|fail(ed|s)? to|did not|does not|"
    r"no evidence|unsupported|refut|challenge[sd]?|in contrast|whereas|myth|"
    r"overstate|overrated|little evidence|mixed evidence|inconsisten)", re.I)


def find_tensions(con, ids, M, topic_query, embedder, pool=80, pairs=15,
                  sim_lo=0.55, sim_hi=0.90):
    """Surface candidate DISAGREEMENTS on a topic.

    Heuristic, not a proof: pull the chunks nearest the topic, keep pairs that
    are similar enough to be about the same thing but not so similar as to be
    duplicates, and rank pairs where at least one side carries contradiction
    cue words. It is a lens for reading, not a verdict. Every hit needs eyes.
    """
    hits = vector_search(topic_query, embedder, ids, M, k=pool)
    idx = {c: i for i, c in enumerate(ids)}
    sel = [c for c, _ in hits if c in idx]
    if len(sel) < 4:
        return []
    sub = M[[idx[c] for c in sel]]
    S = sub @ sub.T
    rows = con.execute("SELECT id, doc_id, text FROM chunks").fetchall()
    txt = {r["id"]: r["text"] for r in rows}
    doc = {r["id"]: r["doc_id"] for r in rows}
    cands = []
    for a in range(len(sel)):
        for b in range(a + 1, len(sel)):
            # Two chunks of the same document are one author restating
            # themselves, not two sources disagreeing. Cross-document only.
            if doc.get(sel[a]) == doc.get(sel[b]):
                continue
            s = float(S[a, b])
            if not (sim_lo <= s <= sim_hi):
                continue
            ta, tb = txt.get(sel[a], ""), txt.get(sel[b], "")
            cue = len(CONTRA_CUES.findall(ta)) + len(CONTRA_CUES.findall(tb))
            if cue == 0:
                continue
            cands.append({"sim": round(s, 3), "cue_hits": cue,
                          "a_chunk": sel[a], "b_chunk": sel[b],
                          "a": ta[:600], "b": tb[:600]})
    cands.sort(key=lambda d: (-d["cue_hits"], -d["sim"]))
    for c in cands[:pairs]:
        for side in ("a", "b"):
            r = con.execute("""SELECT d.title,d.url FROM chunks ch
                               JOIN documents d ON d.id=ch.doc_id WHERE ch.id=?""",
                            (c[f"{side}_chunk"],)).fetchone()
            c[f"{side}_source"] = f"{r['title']} — {r['url']}" if r else "?"
    return cands[:pairs]


def gap_report(con, queries: Sequence[str], embedder, ids, M, threshold=0.72):
    """Which of your questions does the corpus NOT actually answer?

    Runs each query against the corpus and reports the best cosine similarity.
    A low ceiling means you have no strong evidence on that question — which is
    a finding, not a failure, and the thing most corpora never tell you.

    ON THE THRESHOLD. bge-* embeddings are not centred, so every similarity
    sits high: an unrelated question still scores about 0.5 to 0.65 against
    the best chunk. The right threshold depends on the corpus and the model.
    On an earlier private corpus (155 documents, bge-small-en-v1.5) the
    separation fell at about 0.72, which is the default here. On the public
    study corpus in experiments/ (42 documents, 2,104 chunks) the learned
    threshold was about 0.77 for bge-small and about 0.59 for all-MiniLM-L6-v2,
    and the fixed 0.72 classified only 56% of missing documents correctly.
    Re-derive it with `calibrate_threshold` for your corpus and model, and
    treat the ranking as the durable signal: the lowest scoring questions are
    where the evidence is thinnest, wherever the cut-off lands. See the README
    for the full measurement.
    """
    rows = []
    for q in queries:
        hits = vector_search(q, embedder, ids, M, k=5)
        best = hits[0][1] if hits else 0.0
        rows.append({"query": q, "best_similarity": round(best, 3),
                     "n_above_threshold": sum(1 for _, s in hits if s >= threshold),
                     "verdict": "COVERED" if best >= threshold else
                                ("THIN" if best >= threshold - 0.05 else "GAP")})
    import pandas as pd
    return pd.DataFrame(rows).sort_values("best_similarity")


def calibrate_threshold(con, present: Sequence[str], absent: Sequence[str],
                        embedder, ids, M):
    """Re-derive the gap_report threshold for YOUR corpus and model.

    `present` = questions you know the corpus answers.
    `absent`  = questions you know it does not.
    Returns (suggested_threshold, dataframe). If the two bands overlap, the
    midpoint is reported but the overlap is flagged — an overlapping sample
    means the threshold cannot separate them and the ranking is all you have.
    """
    import pandas as pd
    rows = []
    for label, qs in (("present", present), ("absent", absent)):
        for q in qs:
            hits = vector_search(q, embedder, ids, M, k=1)
            rows.append({"label": label, "query": q,
                         "best_similarity": round(hits[0][1] if hits else 0.0, 3)})
    df = pd.DataFrame(rows)
    lo = df[df.label == "present"].best_similarity.min()
    hi = df[df.label == "absent"].best_similarity.max()
    if lo > hi:
        print(f"clean separation: absent tops out at {hi:.3f}, present bottoms at {lo:.3f}")
    else:
        print(f"BANDS OVERLAP: absent reaches {hi:.3f}, present drops to {lo:.3f}. "
              "No threshold separates these — use the ranking, not the verdict.")
    return round((lo + hi) / 2, 3), df.sort_values("best_similarity")


# ────────────────────────────────────────────── grounded answering + audit
ANSWER_SYSTEM = """You are a research analyst working strictly from supplied excerpts.

RULES
1. Every factual sentence ends with a citation marker [n] pointing at the excerpt it came from.
2. If the excerpts do not support a claim, do not make it. Say what is missing instead.
3. Where excerpts disagree, say so explicitly and cite both sides. Do not average them.
4. Separate what the sources state from what you infer. Label inferences INFERENCE.
5. Never cite an excerpt number that was not supplied.
Answer in plain prose. No preamble."""


def build_prompt(question, hits):
    ex = "\n\n".join(
        f"[{i}] SOURCE: {h['title']} ({h['host']})\nURL: {h['url']}\n{h['text'][:1800]}"
        for i, h in enumerate(hits, 1))
    return f"{ANSWER_SYSTEM}\n\nEXCERPTS\n{ex}\n\nQUESTION\n{question}\n\nANSWER"


CITE_RE = re.compile(r"\[(\d{1,2})\]")

def audit_citations(answer: str, hits) -> dict:
    """Verify every [n] in the answer resolves to a supplied excerpt, and report
    which excerpts went unused. This is the cheap version of the citation-
    verification gate — it catches invented markers, not unfaithful ones."""
    used = {int(m) for m in CITE_RE.findall(answer)}
    valid = set(range(1, len(hits) + 1))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    uncited = [s for s in sentences
               if not CITE_RE.search(s) and not s.upper().startswith("INFERENCE")
               and len(s.split()) > 6]
    return {"citations_used": sorted(used),
            "invalid_citations": sorted(used - valid),
            "unused_excerpts": sorted(valid - used),
            "uncited_sentences": uncited,
            "citation_density": round(len(used) / max(len(sentences), 1), 2)}


def make_cards(hits, max_cards=25):
    """Spaced-repetition cards straight from retrieved text, each carrying its
    source. Retrieval practice on primary material beats rereading a summary."""
    cards = []
    for h in hits:
        clean = re.sub(r"\s+", " ", h["text"]).strip()
        for sent in re.split(r"(?<=[.!?])\s+", clean):
            s = sent.strip()
            if 60 < len(s) < 300 and re.search(r"\d|\b(is|are|was|were|found|shows?|reported)\b", s):
                cards.append({"front": f"({h['cluster'] or h['host']}) {h['heading'] or h['title']}: ?",
                              "back": s, "source": h["url"]})
            if len(cards) >= max_cards:
                return cards
    return cards
