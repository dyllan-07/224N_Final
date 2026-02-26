"""
test.py
Load retrieval results, run NLI support scoring over each (query, passage) pair,
and write results with nli_score to a new JSONL file.
"""

import json
import os

from nli import NLISupportScorer
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RETRIEVAL_RESULTS = os.path.join(DATA_DIR, "retrieval_results.jsonl")
NLI_RESULTS_OUT = os.path.join(DATA_DIR, "nli_results.jsonl")
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
    if not os.path.exists(RETRIEVAL_RESULTS):
        raise FileNotFoundError(
            f"Retrieval results not found at {RETRIEVAL_RESULTS}. Run retrieve.py first."
        )

    results = load_retrieval_results(RETRIEVAL_RESULTS)
    if not results:
        print("No retrieval results to score.")
        return

    scorer = NLISupportScorer()
    nli_scores = []

    for i in tqdm(range(0, len(results), BATCH_SIZE), desc="NLI scoring"):
        batch = results[i : i + BATCH_SIZE]
        pairs = [(r["query"], r["passage_text"]) for r in batch]
        scores = scorer.score_batch(pairs)
        nli_scores.extend(scores)

    for r, score in zip(results, nli_scores):
        r["nli_score"] = score

    with open(NLI_RESULTS_OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} results with NLI scores to {NLI_RESULTS_OUT}")


if __name__ == "__main__":
    main()
