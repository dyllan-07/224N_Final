"""
preprocess.py
Load Community Notes TSV, filter to helpful notes with cited URLs,
scrape the cited pages, chunk into passages, and save as JSONL for Pyserini.
"""

import json
import os
import re

import pandas as pd
import trafilatura
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_TSV = os.path.join(DATA_DIR, "notes-small.reduced.tsv")
NOTES_OUT = os.path.join(DATA_DIR, "notes_filtered.parquet")
PASSAGES_DIR = os.path.join(DATA_DIR, "passages")
PASSAGES_OUT = os.path.join(PASSAGES_DIR, "passages.jsonl")

CHUNK_WORDS = 200
CHUNK_OVERLAP = 50


def load_and_filter_notes(path: str) -> pd.DataFrame:
    """Load the TSV and keep only notes that cite at least one URL."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Keep rows where summary contains a URL
    df = df[df["summary"].str.contains(r"https?://", na=False)].copy()
    df = df.dropna(subset=["summary"])
    df = df.reset_index(drop=True)
    print(f"Filtered to {len(df)} notes with URLs")
    return df


def extract_urls(summary: str) -> list[str]:
    """Pull URLs out of the summary text."""
    return re.findall(r"https?://[^\s,\]\)\"]+", str(summary))


def scrape_url(url: str) -> str | None:
    """Fetch and extract main article text from a URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        text = trafilatura.extract(downloaded)
        return text
    except Exception:
        return None


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > chunk_words and current:
            chunks.append(" ".join(current))
            # keep last `overlap` words for continuity
            overlap_words = " ".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)]
            current_len = len(overlap_words)
        current.append(sent)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def main():
    # 1. Load and filter notes
    notes = load_and_filter_notes(RAW_TSV)
    notes.to_parquet(NOTES_OUT, index=False)
    print(f"Saved filtered notes to {NOTES_OUT}")

    # 2. Scrape URLs and chunk into passages
    os.makedirs(PASSAGES_DIR, exist_ok=True)
    passage_id = 0
    with open(PASSAGES_OUT, "w") as f:
        for _, row in tqdm(notes.iterrows(), total=len(notes), desc="Scraping & chunking"):
            urls = extract_urls(row["summary"])
            for url in urls:
                text = scrape_url(url)
                if not text or len(text.split()) < 20:
                    continue
                for chunk in chunk_text(text):
                    record = {
                        "id": str(passage_id),
                        "contents": chunk,
                        "note_id": str(row["noteId"]),
                        "source_url": url,
                    }
                    f.write(json.dumps(record) + "\n")
                    passage_id += 1

    print(f"Saved {passage_id} passages to {PASSAGES_OUT}")


if __name__ == "__main__":
    main()
