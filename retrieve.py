"""
retrieve.py
Build a Pyserini BM25 index over the passage corpus and retrieve
top-k passages for each Community Note query.
"""

import json
import os
import re
import subprocess

import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PASSAGES_JSONL = os.path.join(DATA_DIR, "passages", "passages.jsonl")
INDEX_DIR = os.path.join(DATA_DIR, "bm25_index")
NOTES_PATH = os.path.join(DATA_DIR, "notes_filtered.parquet")
RESULTS_OUT = os.path.join(DATA_DIR, "retrieval_results.jsonl")

TOP_K = 10


def build_index():
    """Build a Lucene index over the passages JSONL using Pyserini."""
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print(f"Index already exists at {INDEX_DIR}, skipping build.")
        return

    # Pyserini expects a directory of JSON files, so we use the JSONL directly
    # via its built-in indexer
    print("Building BM25 index...")
    subprocess.run(
        [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", os.path.dirname(PASSAGES_JSONL),
            "--index", INDEX_DIR,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "4",
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ],
        check=True,
    )
    print(f"Index built at {INDEX_DIR}")


def retrieve(top_k: int = TOP_K):
    """Retrieve top-k passages for each note using BM25."""
    notes = pd.read_parquet(NOTES_PATH)
    searcher = LuceneSearcher(INDEX_DIR)

    results = []
    for _, row in tqdm(notes.iterrows(), total=len(notes), desc="Retrieving"):
        query = re.sub(r"https?://[^\s,\]\)\"]+", "", str(row["summary"])).strip()
        hits = searcher.search(query, k=top_k)

        for rank, hit in enumerate(hits):
            doc = json.loads(hit.lucene_document.get("raw"))
            results.append({
                "note_id": str(row["noteId"]),
                "query": query,
                "rank": rank + 1,
                "passage_id": hit.docid,
                "score": hit.score,
                "passage_text": doc["contents"],
                "source_url": doc.get("source_url", ""),
            })

    with open(RESULTS_OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {RESULTS_OUT}")


def main():
    build_index()
    retrieve()


if __name__ == "__main__":
    main()
