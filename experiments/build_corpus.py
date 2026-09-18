"""Build the study corpus: seed, fetch, chunk and embed with two models.

Run from the repository root:
    uv run --extra fetch --extra embed python experiments/build_corpus.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corpus_brain as cb

DB = "corpus.db"
MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]

con = cb.connect(DB)
rows = list(csv.DictReader(open("seeds.csv")))
print("documents:", cb.seed_documents(con, rows))
cb.fetch_all(con, user_agent="corpus-gaps/0.1 (research on retrieval coverage)")
cb.build_chunks(con)
for name in MODELS:
    cb.embed_chunks(con, cb.Embedder(name, device="cpu"), verbose=False)
    print("embedded with", name)
