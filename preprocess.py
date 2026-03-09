"""
preprocess.py
Load Community Notes TSV, filter to notes with cited URLs,
fetch Wikipedia article text via the MediaWiki API, chunk it,
and save passages as JSONL for Pyserini.
"""

import json
import os
import re
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_RAW_TSV = os.path.join(DATA_DIR, "notes-small.reduced.tsv")
NOTES_OUT = os.path.join(DATA_DIR, "notes_filtered.parquet")
PASSAGES_DIR = os.path.join(DATA_DIR, "passages")
PASSAGES_OUT = os.path.join(PASSAGES_DIR, "passages.jsonl")

CHUNK_WORDS = 200
CHUNK_OVERLAP = 50
MIN_TEXT_WORDS = 20
HTTP_TIMEOUT_SECS = 30
USER_AGENT = "CS224N-Final-Project/1.0 (MediaWiki API preprocessing)"


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


def normalize_wikipedia_host(host: str) -> str:
    """Normalize Wikipedia hosts so mobile URLs use the canonical API host."""
    host = host.lower().split(":", 1)[0]
    return host.replace(".m.wikipedia.org", ".wikipedia.org")


def is_wikipedia_url(url: str) -> bool:
    """Return True if the URL points to a Wikipedia page."""
    host = normalize_wikipedia_host(urlparse(url).netloc)
    return host == "wikipedia.org" or host.endswith(".wikipedia.org")


def wikipedia_api_endpoint(url: str) -> str | None:
    """Build the MediaWiki API endpoint for a Wikipedia article URL."""
    parsed = urlparse(url)
    host = normalize_wikipedia_host(parsed.netloc)
    if not host or not is_wikipedia_url(url):
        return None
    scheme = parsed.scheme or "https"
    return f"{scheme}://{host}/w/api.php"


def wikipedia_title_from_url(url: str) -> str | None:
    """Extract the article title from a standard Wikipedia URL."""
    parsed = urlparse(url)
    if not is_wikipedia_url(url):
        return None

    if parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path[len("/wiki/"):])
    elif parsed.path == "/w/index.php":
        title = parse_qs(parsed.query).get("title", [None])[0]
        title = unquote(title) if title else None
    else:
        return None

    if not title:
        return None

    title = title.split("#", 1)[0].strip().replace("_", " ")
    if not title or title.startswith("Special:"):
        return None
    return title


def fetch_wikipedia_text(url: str) -> str | None:
    """Fetch article text from the MediaWiki API for a Wikipedia URL."""
    endpoint = wikipedia_api_endpoint(url)
    title = wikipedia_title_from_url(url)
    if endpoint is None or title is None:
        return None

    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        }
    )
    request = Request(
        f"{endpoint}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECS) as response:
            payload = json.load(response)
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return None
        text = (pages[0].get("extract") or "").strip()
        return text or None
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
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch cited Wikipedia article text and chunk it into passages"
    )
    parser.add_argument(
        "--notes",
        default=DEFAULT_RAW_TSV,
        help=f"Path to reduced notes TSV (default: {DEFAULT_RAW_TSV})",
    )
    args = parser.parse_args()

    # 1. Load and filter notes
    notes = load_and_filter_notes(args.notes)
    notes.to_parquet(NOTES_OUT, index=False)
    print(f"Saved filtered notes to {NOTES_OUT}")

    # 2. Fetch cited Wikipedia articles and chunk into passages
    os.makedirs(PASSAGES_DIR, exist_ok=True)
    passage_id = 0
    with open(PASSAGES_OUT, "w") as f:
        for _, row in tqdm(notes.iterrows(), total=len(notes), desc="Fetching & chunking"):
            urls = extract_urls(row["summary"])
            for url in urls:
                text = fetch_wikipedia_text(url)
                if not text or len(text.split()) < MIN_TEXT_WORDS:
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
