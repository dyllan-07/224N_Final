"""
evaluate.py
Compute Recall@k and SupportScore@k from an NLI results file.

- Recall@k: proportion of queries where the top-k retrieved sources contain
  at least one gold (cited) source. Gold sources = URLs extracted from the note summary.
- SupportScore@k: mean NLI entailment score over the top-k passages per query
  (how well excerpts support the claim; uses roberta-large-mnli scores already in the file).
"""

import argparse
import json
import os
import re

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NOTES_PATH = os.path.join(DATA_DIR, "notes_filtered.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_NLI_RESULTS = os.path.join(RESULTS_DIR, "bm25_nli_results.jsonl")


def extract_urls(summary: str) -> list[str]:
    """Pull URLs out of the summary text (same logic as preprocess)."""
    return re.findall(r"https?://[^\s,\]\)\"]+", str(summary))


def normalize_url(url: str) -> str:
    """Normalize URL for matching (strip, remove trailing slash)."""
    u = (url or "").strip().rstrip("/")
    return u


def load_gold_sources_by_note(notes_path: str) -> dict[str, set[str]]:
    """Load notes and return for each note_id the set of normalized cited URLs."""
    df = pd.read_parquet(notes_path)
    gold = {}
    for _, row in df.iterrows():
        nid = str(row["noteId"])
        urls = extract_urls(row["summary"])
        gold[nid] = {normalize_url(u) for u in urls}
    return gold


def load_nli_results(path: str) -> list[dict]:
    """Load NLI results JSONL."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def recall_at_k(nli_rows: list[dict], gold_by_note: dict[str, set[str]], k: int) -> float:
    """
    Recall@k = (1/|Q|) * sum_q 1[G(q) ∩ R_k(q) ≠ ∅].
    R_k(q) = set of first k unique source_urls (by rank) for query q.
    """
    from collections import defaultdict
    # Group by note_id, keep rows sorted by rank
    by_note = defaultdict(list)
    for r in nli_rows:
        by_note[r["note_id"]].append(r)
    for nid in by_note:
        by_note[nid].sort(key=lambda x: x["rank"])

    hits = 0
    n_queries = 0
    for nid, rows in by_note.items():
        G = gold_by_note.get(nid)
        if not G:
            continue
        # Build R_k(q): first k unique sources (by order of appearance)
        seen = set()
        R_k = set()
        for r in rows:
            src = normalize_url(r.get("source_url", ""))
            if not src:
                continue
            if src not in seen:
                seen.add(src)
                R_k.add(src)
                if len(R_k) >= k:
                    break
        if G & R_k:
            hits += 1
        n_queries += 1

    return hits / n_queries if n_queries else 0.0


def support_score_at_k(nli_rows: list[dict], k: int) -> float:
    """
    SupportScore@k = (1/|Q|) * sum_q [ mean of nli_score over top-k passages for q ].
    Uses the precomputed nli_score (roberta-large-mnli entailment) in the file.
    """
    from collections import defaultdict
    by_note = defaultdict(list)
    for r in nli_rows:
        by_note[r["note_id"]].append(r)
    for nid in by_note:
        by_note[nid].sort(key=lambda x: x["rank"])

    scores = []
    for nid, rows in by_note.items():
        top_k = [r for r in rows if r["rank"] <= k]
        if not top_k:
            continue
        nli_scores = [r["nli_score"] for r in top_k if "nli_score" in r]
        if not nli_scores:
            continue
        scores.append(sum(nli_scores) / len(nli_scores))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Compute Recall@k and SupportScore@k from NLI results."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_NLI_RESULTS,
        help=f"Path to NLI results JSONL (default: {DEFAULT_NLI_RESULTS})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for Recall@k and SupportScore@k (default: 5)",
    )
    parser.add_argument(
        "--notes",
        default=NOTES_PATH,
        help=f"Path to notes parquet for gold URLs (default: {NOTES_PATH})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"NLI results not found at {args.input}. Run test.py after retrieve.py."
        )
    if not os.path.exists(args.notes):
        raise FileNotFoundError(
            f"Notes not found at {args.notes}. Run preprocess.py first."
        )

    nli_rows = load_nli_results(args.input)
    gold_by_note = load_gold_sources_by_note(args.notes)
    k = args.k

    recall = recall_at_k(nli_rows, gold_by_note, k)
    support = support_score_at_k(nli_rows, k)

    print(f"Recall@{k}\t\t{recall:.4f}")
    print(f"SupportScore@{k}\t{support:.4f}")


if __name__ == "__main__":
    main()
