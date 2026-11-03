# Multi-PDF Semantic RAG System

A Streamlit-based Retrieval-Augmented Generation (RAG) application for asking natural-language questions across multiple PDF documents.

The system extracts text from uploaded PDFs, splits documents into overlapping chunks, generates local semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`, stores them in a ChromaDB vector store, retrieves the most relevant chunks for a user query, and passes the retrieved context to a Groq-hosted Llama model to generate a grounded answer.

---

## Quick Start

> **Live demo:** [https://multi-pdf-semantic-rag.streamlit.app/](https://multi-pdf-semantic-rag.streamlit.app/)
>
> The app is hosted on Streamlit Community Cloud's free tier. If it has not received traffic recently, it may take 20–30 seconds to wake up before accepting uploads.

To run locally, see [Installation](#installation).

---

## Table of Contents

- [Why This Project](#why-this-project)
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Metrics and Experimentation](#metrics-and-experimentation)
- [Experiment Logging](#experiment-logging)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Why This Project

This project was developed for two related reasons.

The first is practical: building a full end-to-end RAG pipeline from scratch, covering document ingestion, chunking, embedding, vector indexing, similarity retrieval, context construction, LLM generation, and latency measurement, is a useful exercise in understanding where things can go wrong in each stage independently.

The second reason is research-oriented. Building the pipeline made it clear that the harder and less-studied problem is not *how* to build a RAG system, but *how to know whether it is working correctly*. Specifically: given a fixed document corpus, how should one systematically test whether the system's answers are correct and grounded in the source material? And how do chunking parameters (chunk size, overlap) and retrieval configuration (top-k) interact to affect answer quality?

This is not only a software engineering question but also a testing and verification question: RAG systems introduce a retrieval step whose correctness cannot be checked by unit tests alone, and an LLM generation step whose outputs are non-deterministic. Designing a principled evaluation methodology for such a system, and identifying where it fails and why, is the open problem this project begins to address.

---

## Overview

This project focuses on building and evaluating an end-to-end semantic RAG pipeline for multi-document question answering.

The system provides:

* Multi-PDF document ingestion (up to 1,600 files, 10,000 MB per session)
* SHA-256 duplicate detection
* Local semantic embeddings (no external embedding API)
* Persistent ChromaDB vector store
* Retrieval-augmented generation with source attribution
* Retrieved passage inspection with similarity scores
* Batch document processing with per-file status reporting
* Structured indexing and retrieval metrics
* JSONL experiment logging

---

## How It Works

```text
                    PDF Documents
                         |
                         v
                  PDF Text Extraction
                   (PyPDFLoader, page-by-page)
                         |
                         v
                  Text Chunking
              (1000 chars, 200 overlap,
               RecursiveCharacterTextSplitter)
                         |
                         v
             Sentence Transformer
              Embedding Generation
           (all-MiniLM-L6-v2, 384-dim,
            L2-normalised, CPU inference)
                         |
                         v
                  ChromaDB (persistent,
                  cosine similarity via
                  normalised L2 distance)
                         |
                         |
User Question ----------+
                         |
                         v
                  Query Embedding
                         |
                         v
              Similarity Search
                    (Top-K = 6)
                         |
                         v
                Retrieved Context
                         |
                         v
                  Groq Llama LLM
               (llama-3.1-8b-instant)
                         |
                         v
                Grounded Answer
                         |
                         v
             Sources + Retrieved Chunks
```

### Pipeline

1. **Ingestion**

   Uploaded PDFs are fingerprinted using SHA-256. Duplicate files are detected before any processing occurs and skipped with a log entry. Each file's fingerprint, name, size, chunk count, and index timestamp are written to a per-session `manifest.json`.

2. **Text Extraction**

   PDF content is extracted page by page using `PyPDFLoader` from `langchain-community`. PDFs that yield no text are marked as `empty` and skipped.

3. **Chunking**

   Extracted text is split into overlapping chunks using `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", "! ", "? ", " "]` and `keep_separator=True`. Default chunk size is 1,000 characters with 200-character overlap.

4. **Embedding**

   Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors) with `normalize_embeddings=True`. Embeddings are cached on disk via `CacheBackedEmbeddings` with SHA-256 keying to avoid re-embedding unchanged chunks across sessions.

5. **Vector Storage**

   Embeddings and chunk metadata are stored in a **ChromaDB** persistent store (`chromadb==1.5.9`). Each session uses a separate on-disk collection. The ChromaDB client is opened and explicitly released within a context manager for each operation — no client object is held across Streamlit reruns.

6. **Retrieval**

   The query is embedded using the same model. ChromaDB's `similarity_search_with_relevance_scores` retrieves the top-k most similar chunks. Because embeddings are L2-normalised, the underlying L2 distance is equivalent to cosine similarity. The default `k` is **6**.

7. **Generation**

   Retrieved chunks are assembled into a structured prompt. The prompt instructs the model to answer using only the retrieved context, not to invent technical identifiers, and to report conflicting information verbatim. The Groq-hosted `llama-3.1-8b-instant` model generates the response. LLM responses are cached in a local SQLite database via `SQLiteCache`.

8. **Presentation**

   The application displays the generated answer, source PDFs, retrieval timing, and the full retrieved passages with source file, page number, chunk ID, similarity score, and character count.

---

## Features

### Multi-PDF Support

Upload and process multiple PDF documents within a single session. Each session maintains its own ChromaDB collection and manifest.

### Local Embeddings

Uses `sentence-transformers/all-MiniLM-L6-v2` for local CPU embedding generation. No external embedding API or key is required.

### ChromaDB Persistent Vector Store

Embeddings are persisted to disk in a session-scoped ChromaDB directory. The store survives application restarts and supports incremental indexing — new files can be added to an existing session without re-indexing previously processed documents.

### Semantic Retrieval

Queries are matched against indexed chunks using cosine similarity (via L2-normalised embeddings in ChromaDB). Each retrieved chunk is returned with its relevance score.

### Duplicate Detection

Uploaded files are fingerprinted using SHA-256 before any processing. If a file with the same fingerprint has already been indexed in the current session, it is skipped. This is checked against the persisted manifest, so it survives application restarts.

### Batch Processing

Chunks are inserted into ChromaDB in configurable batches (default: 100 chunks per batch) to limit peak RAM usage. Each file is processed and its temp file deleted before the next file begins.

### Source-Aware Answers

Retrieved chunks carry `source_pdf`, `page`, `chunk_id`, `chunk_index`, `total_chunks`, and `char_count` metadata. Source PDFs are listed alongside each answer.

### Retrieval Transparency

Users can expand a "Retrieved Chunks" panel to inspect every passage used by the model, along with its similarity score and origin.

### Experiment Metrics

Every indexing and retrieval operation records timing and size metrics, written to `experiment_runs.jsonl` for offline analysis.

---

## Evaluation

> **Status:** The evaluation infrastructure (`evaluation.py`, `metrics.py`, `experiment_runs.jsonl`) is in place. Formal evaluation runs have not yet been conducted. All result cells below are marked `TBD` and will be filled as experiments are completed.

### Methodology

#### Question Set

Questions will be written manually against a fixed document corpus. Ground-truth answers will be derived by reading the source PDFs directly, not by asking the system. Questions will cover three difficulty levels:

- **Lookup**: single-chunk, single-document answers (e.g., a specific named entity, date, or CVE)
- **Synthesis**: answers that require combining information from multiple chunks or multiple documents
- **Negative**: questions whose answers are not present in the corpus (expected system response: *"Information not found in retrieved documents."*)

#### Retrieval Hit Rate

For each question, the ground-truth source chunk (identified by fingerprint and page number) will be checked against the top-k retrieved chunks. A retrieval is counted as a hit if the ground-truth chunk appears in the retrieved set.

#### Answer Correctness Scoring

Answers will be scored on a 0–2 scale per question:

| Score | Meaning |
|-------|---------|
| 2 | Fully correct and grounded in the retrieved context |
| 1 | Partially correct or missing relevant detail present in context |
| 0 | Incorrect, hallucinated, or refused despite relevant context being present |

Scoring will be performed manually by the author against the ground-truth answers.

#### Faithfulness Check

For each answer, the retrieved chunks used to generate it will be inspected to determine whether every factual claim in the answer is directly traceable to the retrieved text. Answers containing claims not present in any retrieved chunk will be flagged as unfaithful.

> **Note on automation:** `evaluation.py` currently contains a placeholder dataclass (`ExperimentSnapshot`) with a note referencing future RAGAS, retriever, and embedding evaluations. No automated scoring is implemented yet.

### Results

| Configuration | Chunk Size | Overlap | Top-K | Questions | Retrieval Hit Rate | Answer Correctness (avg) | Faithfulness |
|---------------|-----------|---------|-------|-----------|-------------------|--------------------------|-------------|
| Baseline      | 1000      | 200     | 6     | TBD       | TBD               | TBD                      | TBD         |
| Larger chunks | TBD       | TBD     | 6     | TBD       | TBD               | TBD                      | TBD         |
| Higher k      | 1000      | 200     | TBD   | TBD       | TBD               | TBD                      | TBD         |

### Failure Cases

The following are placeholder entries in the format used to document observed failures. They will be filled in during evaluation runs.

| # | Question | Expected Answer | System Returned | Likely Cause |
|---|----------|----------------|-----------------|--------------|
| 1 | TBD | TBD | TBD | Chunking — answer spans a chunk boundary and is split across two non-retrieved chunks |
| 2 | TBD | TBD | TBD | Retrieval — semantically similar but topically irrelevant chunks ranked above the correct one |
| 3 | TBD | TBD | TBD | Generation — model paraphrases or omits specific technical identifiers present in the retrieved text |

### Threats to Validity

- **Small question set.** A small manually written question set may not cover the full range of query types or document topics. Results may not generalise.
- **Author-written questions.** Questions and ground-truth answers are written by the same person who built the system. This introduces the risk of unconsciously favouring query phrasings that the system handles well.
- **LLM non-determinism.** `llama-3.1-8b-instant` is called via the Groq API without fixing a random seed (the API does not expose one). The same question may receive different answers across runs. LLM response caching (`SQLiteCache`) mitigates this for repeated identical prompts but does not eliminate variability for new queries.
- **LLM judge variability.** If automated LLM-based scoring is used in future, the judge model introduces its own error and bias.

---

## Project Structure

| File               | Responsibility                                                |
| ------------------ | ------------------------------------------------------------- |
| `app.py`           | Streamlit interface and application orchestration             |
| `config.py`        | Central application configuration                             |
| `ingestion.py`     | PDF loading, deduplication, metadata tagging and indexing     |
| `chunking.py`      | Text splitter configuration                                   |
| `embeddings.py`    | Embedding model loading and disk-backed cache                 |
| `vectordb.py`      | ChromaDB lifecycle, session management, manifest persistence  |
| `retrieval.py`     | Similarity search, context construction and prompt generation |
| `llm.py`           | Groq LLM configuration and SQLite response cache             |
| `metrics.py`       | Indexing and query metric dataclasses                         |
| `evaluation.py`    | Evaluation scaffold (in progress)                             |
| `logging_utils.py` | Application logging and JSONL experiment record appending     |
| `requirements.txt` | Python dependencies                                           |
| `README.md`        | Project documentation                                         |

---

## Tech Stack

* **Python 3.10+**
* **Streamlit** — interactive web interface
* **LangChain** (`langchain`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`, `langchain-classic`) — pipeline orchestration
* **sentence-transformers / all-MiniLM-L6-v2** — local CPU embedding model (384-dim)
* **ChromaDB 1.5.9** — persistent vector store (cosine similarity via normalised L2)
* **Groq / llama-3.1-8b-instant** — hosted LLM for answer generation
* **PyPDF** — PDF text extraction

---

## Requirements

* Python 3.10+
* Groq API key (free tier available at [console.groq.com](https://console.groq.com))
* Disk space for the embedding model (~90 MB) and ChromaDB collections
* Internet connection for package installation and the Groq API

---

## Installation

```bash
git clone https://github.com/swahathi/vector_rag.git
cd vector_rag
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

The application requires a Groq API key for question answering. Embedding and retrieval work without it; only the LLM generation step will fail.

For local execution, set the environment variable before running:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-api-key"

# Linux/macOS
export GROQ_API_KEY="your-api-key"
```

For Streamlit Community Cloud deployment, configure `GROQ_API_KEY` through the application's Streamlit Secrets settings.

The API key should never be committed to the repository.

All other parameters (chunk size, overlap, top-k, batch size, model names, file limits) are set in [`config.py`](config.py) and require no environment variables.

---

## Usage

Run the application locally:

```bash
streamlit run app.py
```

Then:

1. Upload one or more PDF documents using the file uploader.
2. Click **Process PDFs**.
3. Wait for the indexing process to complete. Per-file status (indexed / duplicate / empty / failed) and timing metrics are displayed after processing.
4. Enter a natural-language question in the text input.
5. Click **Get Answer**.
6. Review the generated answer and the list of source PDFs.
7. Expand the **Retrieved Chunks** panel to inspect the passages used by the model, including similarity scores and page numbers.

Additional PDFs can be uploaded and indexed incrementally during the session without re-indexing previously processed documents.

Use **Clear My Data** in the sidebar to delete the session's ChromaDB collection and manifest and start fresh.

---

## Metrics and Experimentation

The application records metrics for every indexing and retrieval operation.

### Indexing Metrics

| Metric | Description |
|--------|------------|
| `pdf_count` | Number of PDFs submitted |
| `dataset_size_mb` | Total upload size |
| `load_time` | Time spent in PyPDFLoader across all files |
| `chunk_time` | Time spent in the text splitter |
| `embedding_time` | Time spent embedding and inserting chunks |
| `vector_insertion_time` | Time spent in ChromaDB `add_documents` calls |
| `chroma_init_time` | Time to open the ChromaDB client |
| `total_indexing_time` | Wall-clock time for the full indexing run |
| `total_chunks` | Chunks inserted in this run |
| `processed_files` | Files successfully indexed |
| `skipped_duplicates` | Files skipped due to SHA-256 match |
| `failed_files` | Files that raised an exception |
| `db_size_mb` | On-disk size of the ChromaDB collection after indexing |
| `chunk_size` / `chunk_overlap` | Configuration used for this run |

### Query Metrics

| Metric | Description |
|--------|------------|
| `chroma_retrieval_time` | Time for the ChromaDB similarity search call |
| `retrieval_time` | Total retrieval wall-clock time |

These metrics are displayed in the UI after each operation and also written to `experiment_runs.jsonl`, making it straightforward to compare runs with different chunk sizes, overlaps, or top-k values using Python or a spreadsheet.

---

## Experiment Logging

The application maintains two log files:

**`rag.log`** — human-readable log of application events, including per-file indexing results, skipped duplicates, warnings from PyPDF, and query timestamps.

**`experiment_runs.jsonl`** — one JSON record per indexing run, containing the full `IndexingMetrics` dictionary and a session ID. Records can be loaded with:

```python
import json, pandas as pd

runs = [json.loads(line) for line in open("experiment_runs.jsonl")]
df = pd.json_normalize([r["metrics"] for r in runs])
print(df[["chunk_size", "chunk_overlap", "total_chunks", "embedding_time", "total_indexing_time"]])
```

---

## Limitations

* Only text-based PDFs are supported. Scanned PDFs without a text layer require OCR, which is not implemented.
* Retrieval quality depends on the embedding model, chunking strategy, and top-k value. No automated retrieval evaluation is currently in place.
* Answer quality depends on the retrieved context and the LLM. The model is instructed not to invent information, but LLM non-determinism and prompt sensitivity mean faithfulness is not guaranteed.
* The evaluation framework (`evaluation.py`) is currently a stub. Systematic evaluation has not yet been performed.
* Streamlit Community Cloud's free tier has memory and CPU constraints that limit the size of document collections that can be processed in the deployed environment.
* API-based LLM generation requires a valid Groq API key and an active network connection.

---

## Future Improvements

### Research-Oriented

* **Mutation-style retrieval testing.** Systematically perturb queries (synonym substitution, negation, entity swaps) and test whether the system's answer changes in the expected direction. This is analogous to mutation testing in traditional software testing.
* **Consistency testing across paraphrased questions.** The same factual question expressed differently should produce the same answer. Measuring consistency across paraphrase sets is a tractable proxy for correctness that does not require a ground-truth oracle.
* **Hallucination detection.** Implement a faithfulness check that verifies every factual claim in a generated answer against the retrieved context. This could use a secondary LLM call (NLI-style entailment) or an exact-match scan for specific entity types.
* **Automated evaluation pipeline.** Replace the current manual scoring approach with a RAGAS-style automated evaluation covering retrieval hit rate, answer faithfulness, and answer relevance, using a fixed question-answer test set.

### Engineering

* Hybrid keyword + semantic retrieval (BM25 + vector, with score fusion)
* Reranking retrieved chunks with a cross-encoder before passing to the LLM
* Support for scanned PDFs via OCR
* Metadata filtering to restrict retrieval to a subset of indexed documents
* Multiple embedding model comparisons with shared evaluation infrastructure
* Improved multi-user session isolation

---

## Why This Project

This project was developed for two related reasons.

The first is practical: building a complete RAG pipeline from scratch, rather than using an off-the-shelf framework, forces engagement with each stage individually — chunking decisions, embedding normalisation, ChromaDB client lifecycle management, prompt construction — and makes it easier to reason about where failures originate.

The second is research-oriented. Building the pipeline made it clear that the harder problem is not *how* to build a RAG system but *how to know whether it is working correctly*. Specifically: given a fixed document corpus, how should one systematically test whether the system's answers are correct and grounded in the source material? And how do chunking parameters and retrieval depth interact to affect answer quality?

These are open questions at the intersection of software testing, formal grounding of LLM outputs, and information retrieval evaluation. This project is an initial step toward addressing them practically.

---

## GitHub Repository Settings

> **Suggested one-line description** (paste into the repo's "About" field on GitHub):
>
> *Multi-PDF semantic RAG system built with Streamlit, LangChain, ChromaDB, and Groq — with a focus on retrieval correctness and evaluation methodology.*

> **Suggested topics:**
> `rag` `llm` `langchain` `chromadb` `retrieval-augmented-generation` `streamlit` `sentence-transformers` `information-retrieval`

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

The deployed application requires `GROQ_API_KEY` to be configured through Streamlit Secrets. Embedding and document indexing work without it; question answering will fail until the key is set.

---

## Contributing

Issues and pull requests are welcome.

For significant changes, open an issue first to discuss the proposed approach.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Swathy M
