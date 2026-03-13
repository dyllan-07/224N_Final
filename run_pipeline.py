"""
run_pipeline.py
Single-command runner for the full RAG pipeline: preprocess → retrieve → NLI → evaluate.
Supports --modes (bm25 and/or hybrid), --skip (default True), and --k for evaluation.
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DEFAULT_NOTES_TSV = os.path.join(DATA_DIR, "notes-small.reduced.tsv")
PASSAGES_JSONL = os.path.join(DATA_DIR, "passages", "passages.jsonl")
NOTES_PARQUET = os.path.join(DATA_DIR, "notes_filtered.parquet")

VALID_MODES = {"bm25", "hybrid", "hybrid-rerank"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: preprocess → retrieve → NLI → evaluate."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(VALID_MODES),
        default=["bm25"],
        metavar="MODE",
        help="Retrieval modes to run: bm25, hybrid, or both (default: bm25)",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        default=True,
        help="Skip a step if its output file(s) already exist (default: True)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_false",
        dest="skip",
        help="Run all steps even if output files exist",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for Recall@k and SupportScore@k in evaluation (default: 5)",
    )
    parser.add_argument(
        "--notes",
        default=DEFAULT_NOTES_TSV,
        help=f"Path to reduced notes TSV (default: notes-small.reduced.tsv; use notes-large-helpful.reduced.tsv for helpful-only set)",
    )
    return parser.parse_args()


def validate_inputs(notes_path: str):
    """Reject if required input TSV is missing."""
    if not os.path.isfile(notes_path):
        print(f"Error: input notes file not found: {notes_path}", file=sys.stderr)
        print("Create it with: python reduce_tsv.py data/notes.tsv -o data/notes-small.reduced.tsv", file=sys.stderr)
        print("Or for helpful notes: python data/recentHelpful.py (after notes-large.reduced.tsv exists)", file=sys.stderr)
        sys.exit(1)


def run(cmd: list[str], step_name: str) -> bool:
    """Run a command; return True on success, False on failure."""
    print(f"[run_pipeline] {step_name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Error: {step_name} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return True


def main():
    args = parse_args()
    notes_path = args.notes

    validate_inputs(notes_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Preprocess (force run if using a different notes file so we don't reuse stale passages)
    skip_preprocess = (
        args.skip
        and notes_path == DEFAULT_NOTES_TSV
        and os.path.isfile(PASSAGES_JSONL)
        and os.path.isfile(NOTES_PARQUET)
    )
    if skip_preprocess:
        print("[run_pipeline] Preprocess: skipped (outputs already exist)")
    else:
        run([sys.executable, "preprocess.py", "--notes", notes_path], "Preprocess")

    # 2. Retrieve (per mode)
    for mode in args.modes:
        retrieve_mode = "hybrid" if mode == "hybrid-rerank" else mode
        top_k = 50 if mode == "hybrid-rerank" else 10
        results_file = os.path.join(RESULTS_DIR, f"{retrieve_mode}_results.jsonl")
        skip_retrieve = args.skip and os.path.isfile(results_file)
        if skip_retrieve:
            print(f"[run_pipeline] Retrieve ({mode}): skipped ({results_file} exists)")
        else:
            run(
                [sys.executable, "retrieve.py", "--mode", retrieve_mode, "--top-k", str(top_k)],
                f"Retrieve ({mode})",
            )

    # 2b. Rerank (hybrid-rerank mode only)
    for mode in args.modes:
        if mode != "hybrid-rerank":
            continue
        reranked_file = os.path.join(RESULTS_DIR, "hybrid_reranked_results.jsonl")
        hybrid_file = os.path.join(RESULTS_DIR, "hybrid_results.jsonl")
        skip_rerank = args.skip and os.path.isfile(reranked_file)
        if skip_rerank:
            print(f"[run_pipeline] Rerank: skipped ({reranked_file} exists)")
        else:
            run(
                [sys.executable, "rerank.py", "-i", hybrid_file],
                "Rerank",
            )

    # 3. NLI (per mode)
    for mode in args.modes:
        if mode == "hybrid-rerank":
            retrieval_file = os.path.join(RESULTS_DIR, "hybrid_reranked_results.jsonl")
            nli_file = os.path.join(RESULTS_DIR, "hybrid_reranked_nli_results.jsonl")
        else:
            retrieval_file = os.path.join(RESULTS_DIR, f"{mode}_results.jsonl")
            nli_file = os.path.join(RESULTS_DIR, f"{mode}_nli_results.jsonl")
        skip_nli = args.skip and os.path.isfile(nli_file)
        if not os.path.isfile(retrieval_file):
            print(f"[run_pipeline] NLI ({mode}): skipped (no retrieval results: {retrieval_file})")
            continue
        if skip_nli:
            print(f"[run_pipeline] NLI ({mode}): skipped ({nli_file} exists)")
        else:
            nli_cmd = [sys.executable, "test.py", "-i", retrieval_file]
            if mode == "hybrid":
                nli_cmd += ["--top-k", "10"]
            run(nli_cmd, f"NLI ({mode})")

    # 4. Evaluate (per mode)
    for mode in args.modes:
        if mode == "hybrid-rerank":
            nli_file = os.path.join(RESULTS_DIR, "hybrid_reranked_nli_results.jsonl")
        else:
            nli_file = os.path.join(RESULTS_DIR, f"{mode}_nli_results.jsonl")
        if not os.path.isfile(nli_file):
            print(f"[run_pipeline] Evaluate ({mode}): skipped (no NLI results: {nli_file})")
            continue
        run(
            [sys.executable, "evaluate.py", "-i", nli_file, "--k", str(args.k)],
            f"Evaluate ({mode})",
        )


if __name__ == "__main__":
    main()
