# 224N Final — Retrieval for Community Notes

Retrieve and score source passages for Community Notes using BM25 and/or hybrid (BM25 + dense) retrieval, then evaluate with NLI-based support scoring.

## Setup

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Java 11+** (required by Pyserini for the BM25 index)
   - **Apple Silicon Mac**: Use an ARM64 JDK (e.g. [Eclipse Temurin](https://adoptium.net/) or `conda install -c conda-forge openjdk=21`). Then:
     ```bash
     export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home
     export PATH="$JAVA_HOME/bin:$PATH"
     ```
   - **Intel Mac / Linux**: Install via your package manager (e.g. `brew install openjdk@21`).

## How to run the pipeline

Run these steps in order. Inputs and outputs are under `data/` and `results/`.

| Step | Command | What it does |
|------|--------|----------------|
| **1. Preprocess** | `python preprocess.py` | Reads `data/notes-small.reduced.tsv`, scrapes cited URLs, chunks text → `data/passages/passages.jsonl` and `data/notes_filtered.parquet`. |
| **2. Retrieve** | `python retrieve.py --mode bm25` | Builds BM25 index (if needed), retrieves top-10 passages per note → `results/bm25_results.jsonl`. |
| | `python retrieve.py --mode hybrid` | Same, but BM25 + dense retrieval with RRF merge → `results/hybrid_results.jsonl`. |
| **3. NLI scoring** | `python test.py` | Scores each (query, passage) with RoBERTa-large-MNLI → adds `nli_score` to each row. Default input: `results/bm25_results.jsonl` → output: `results/bm25_nli_results.jsonl`. |
| | `python test.py -i results/hybrid_results.jsonl` | Same for hybrid results → `results/hybrid_nli_results.jsonl`. |
| **4. Evaluate** | `python evaluate.py` | Computes **Recall@k** and **SupportScore@k**. Default: reads `results/bm25_nli_results.jsonl`, uses `--k 5`. Prints the two scores. |
| | `python evaluate.py -i results/hybrid_nli_results.jsonl --k 10` | Evaluate hybrid NLI results with k=10. |

### Quick copy-paste

**BM25 only:**
```bash
python preprocess.py
python retrieve.py --mode bm25
python test.py
python evaluate.py
```

**Hybrid (BM25 + dense):**
```bash
python preprocess.py
python retrieve.py --mode hybrid
python test.py -i results/hybrid_results.jsonl
python evaluate.py -i results/hybrid_nli_results.jsonl
```

### Evaluation metrics

- **Recall@k**: Fraction of notes where at least one of the top-k *sources* returned is a cited (gold) URL. Gold URLs come from the note’s `summary`; `evaluate.py` reads `data/notes_filtered.parquet` to get them.
- **SupportScore@k**: Mean NLI entailment score over the top-k passages per note (how well excerpts support the claim), using the `nli_score` values already in the NLI results file.

Optional args: `evaluate.py --input <path> --k <int> --notes <parquet>`.

## Data and results layout

| Path | Description |
|------|-------------|
| `data/notes-small.reduced.tsv` | Input notes (e.g. 101 rows with Wikipedia/cited URLs). |
| `data/notes_filtered.parquet` | Notes used by retrieval + evaluation (from preprocess). |
| `data/passages/passages.jsonl` | Chunked passages from scraped URLs. |
| `data/bm25_index/` | Lucene index (built by retrieve). |
| `results/bm25_results.jsonl` | BM25 retrieval output (passage-level). |
| `results/hybrid_results.jsonl` | Hybrid retrieval output. |
| `results/*_nli_results.jsonl` | Retrieval results + `nli_score` (from test.py). |

## Utility

Reduce a raw Community Notes TSV to rows with Wikipedia links and key columns:
```bash
python reduce_tsv.py data/notes.tsv -o data/notes.reduced.tsv
```

## Troubleshooting

- **SIGBUS / Java crash on Apple Silicon**: If `retrieve.py` crashes with a JVM CodeHeap error, try Eclipse Temurin (ARM64) and ensure `JAVA_HOME` and `PATH` point to it. If it still fails, see in-repo notes on running under Rosetta or building the index on another machine.
