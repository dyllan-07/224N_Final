"""
test.py
Load retrieval results, run NLI support scoring over each (query, passage) pair,
and write results with nli_score to a new JSONL file.
"""

import argparse
import json
import os

from nli import NLISupportScorer
from tqdm import tqdm


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_RETRIEVAL_RESULTS = os.path.join(RESULTS_DIR, "bm25_results.jsonl")
BATCH_SIZE = 32


def load_retrieval_results(path: str) -> list:
    """Load retrieval_results.jsonl into a list of dicts."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run NLI support scoring on retrieval results")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_RETRIEVAL_RESULTS,
        help=f"Path to retrieval results JSONL (default: {DEFAULT_RETRIEVAL_RESULTS})",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for NLI-scored output JSONL (default: results/nli_results.jsonl or derived from input name)",
    )
    args = parser.parse_args()

    retrieval_path = args.input
    if not os.path.exists(retrieval_path):
        raise FileNotFoundError(
            f"Retrieval results not found at {retrieval_path}. Run retrieve.py first."
        )

    results = load_retrieval_results(retrieval_path)
    if not results:
        print("No retrieval results to score.")
        return

    if args.output:
        nli_out = args.output
    else:
        base = os.path.splitext(os.path.basename(retrieval_path))[0]
        nli_out = os.path.join(RESULTS_DIR, f"{base.replace('_results', '_nli_results')}.jsonl")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    scorer = NLISupportScorer()
    nli_scores = []

    for i in tqdm(range(0, len(results), BATCH_SIZE), desc="NLI scoring"):
        batch = results[i : i + BATCH_SIZE]
        pairs = [(r["query"], r["passage_text"]) for r in batch]
        scores = scorer.score_batch(pairs)
        nli_scores.extend(scores)

    for r, score in zip(results, nli_scores):
        r["nli_score"] = score

    with open(nli_out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} results with NLI scores to {nli_out}")


if __name__ == "__main__":
    main()
