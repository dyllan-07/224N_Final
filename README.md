# 224N Final — BM25 Retrieval for Community Notes

## Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Java 11+** (required by Pyserini):
   - **Apple Silicon Mac**: Install an ARM64 JDK via conda:
     ```bash
     conda install -c conda-forge openjdk=21
     ```
     Then set the env var before running retrieval:
     ```bash
     export JAVA_HOME=/opt/miniconda3/lib/jvm
     ```
   - **Intel Mac / Linux**: Any Java 11+ works. Install via your package manager (e.g. `brew install openjdk@21` or `apt install openjdk-21-jdk`).

## Running

```bash
# Step 1: Scrape URLs from notes, chunk into passages
python preprocess.py

# Step 2: Build BM25 index and retrieve top-10 passages per note
python retrieve.py
```

- `preprocess.py` reads `data/notes-small.reduced.tsv`, scrapes cited URLs, and writes passages to `data/passages/passages.jsonl`.
- `retrieve.py` builds a Lucene index over the passages and saves retrieval results to `data/retrieval_results.jsonl`.
