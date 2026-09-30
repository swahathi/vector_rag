# Multi-PDF Semantic RAG System

A Streamlit application for asking natural-language questions across multiple PDF documents. Uploaded PDFs are chunked, embedded locally with a sentence-transformer model, and stored in a per-session Chroma vector database. At query time, the most relevant chunks are retrieved and passed to a Groq-hosted Llama model, which produces an answer grounded in your documents along with the source PDFs and retrieved passages.

The app also reports detailed timing and size metrics for every indexing run and query, which makes it useful for experimenting with chunking and retrieval settings.

## Features

- Upload and index many PDFs in a single session
- Local embeddings with `sentence-transformers/all-MiniLM-L6-v2` (no embedding API key required)
- Persistent, session-scoped Chroma vector store
- Duplicate detection using SHA-256 file fingerprints, so re-uploading the same file is skipped
- Batched ingestion with per-file status reporting (indexed, duplicate, empty, failed)
- Answer generation with Groq (`llama-3.1-8b-instant`) through LangChain
- Transparent results: answer, source PDFs, and an expandable view of retrieved chunks with page number, chunk ID, similarity score, and character count
- Experiment metrics for each run: Chroma initialization, PDF loading, chunking, embedding, vector insertion, total time, and database size
- Structured logging to a log file and a JSONL experiment log
- One-click "Clear My Data" to delete the session's vector database
- Render deployment configuration included

## How It Works

```
PDF upload
    |
    v
PyPDFLoader -> text splitter (1000 chars, 200 overlap)
    |
    v
HuggingFace embeddings (all-MiniLM-L6-v2)
    |
    v
Chroma vector store (one per session, with a manifest of indexed files)
    |
    v
Query -> top-k similarity search (k = 20) -> context + prompt -> Groq LLM -> answer + sources
```

1. **Ingestion.** Each uploaded PDF is fingerprinted with SHA-256. Files already present in the session manifest are skipped. New files are written to a temporary location, loaded page by page, split into overlapping chunks, and tagged with metadata (source PDF, page, chunk ID, chunk index, total chunks, character count, session ID, file fingerprint).
2. **Indexing.** Chunks are embedded and inserted into Chroma in batches. A manifest records every indexed file and every run.
3. **Retrieval.** The question is embedded and the top `k` most similar chunks are retrieved with scores.
4. **Generation.** Retrieved chunks are assembled into a context block and combined with the question in a prompt sent to the Groq LLM.
5. **Presentation.** The app shows the answer, the list of source PDFs, retrieval metrics, and the raw retrieved chunks.

## Project Structure

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI and pipeline orchestration (upload, process, ask, display) |
| `config.py` | Central, immutable configuration (`AppConfig`) |
| `ingestion.py` | PDF loading, deduplication, metadata tagging, batched indexing |
| `chunking.py` | Text splitter construction |
| `embeddings.py` | Embedding model loading |
| `vectordb.py` | Session-scoped Chroma lifecycle, manifest handling, size reporting, cleanup |
| `retrieval.py` | Similarity search, context building, prompt construction |
| `llm.py` | Groq LLM loading |
| `metrics.py` | Indexing and query metric data structures |
| `evaluation.py` | Experiment snapshot support |
| `logging_utils.py` | Log configuration and JSONL record writer |
| `render.yaml` | Render deployment configuration |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10 or newer
- A Groq API key ([console.groq.com](https://console.groq.com))
- Enough disk space for the embedding model download and the Chroma database

## Installation

```bash
git clone https://github.com/swahathi/vector_rag.git
cd vector_rag

python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

Set your Groq API key as an environment variable before starting the app.

```bash
export GROQ_API_KEY="your-api-key"        # PowerShell: $env:GROQ_API_KEY="your-api-key"
```

The app displays a warning in the UI if `GROQ_API_KEY` is missing; indexing still works, but question answering will fail.

All tunable settings live in `config.py`:

| Setting | Default | Description |
| --- | --- | --- |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `200` | Overlapping characters between adjacent chunks |
| `chunk_batch_size` | `1000` | Chunks embedded and inserted per batch |
| `retrieval_k` | `20` | Number of chunks retrieved per query |
| `embedding_model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `groq_model_name` | `llama-3.1-8b-instant` | Groq model used for answers |
| `max_files` | `1600` | Maximum files per upload |
| `max_total_mb` | `10000` | Maximum total upload size in MB |
| `vector_db_root` | `./chroma_sessions` | Directory for session vector databases |
| `collection_name` | `rag_collection` | Chroma collection name |
| `embed_cache_dir` | `./embedding_cache` | Embedding cache directory |
| `llm_cache_file` | `./langchain_cache.db` | LangChain LLM cache |
| `log_file` | `./rag.log` | Application log |
| `experiment_log_file` | `./experiment_runs.jsonl` | One JSON record per indexing run |

## Usage

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`) and:

1. Upload one or more PDFs with the file uploader.
2. Click **Process PDFs** and wait for indexing to finish. A metrics panel summarizes the run.
3. Type a question under **Ask a question** and click **Get Answer**.
4. Review the answer, the source PDFs, and, if you wish, the retrieved chunks.

You can upload additional PDFs at any time; already-indexed files are detected and skipped. Use **Clear My Data** in the sidebar to delete the session's vector database and start fresh.

## Metrics and Logging

**Indexing metrics** (shown after each Process PDFs run): Chroma initialization time, PDF loading time, chunking time, embedding time, vector insertion time, total processing time, vector database size, total chunks, PDF count, dataset size, and the chunk size and overlap used.

**Query metrics**: Chroma retrieval time, overall retrieval time, and the configured chunk size.

**Logs**:

- `rag.log` records pipeline steps, successes, failures, and user queries.
- `experiment_runs.jsonl` appends one JSON record per indexing run (timestamp, session ID, and full metrics), ready for analysis in pandas or a spreadsheet.

Together these make it straightforward to compare the effect of different `chunk_size`, `chunk_overlap`, `retrieval_k`, or embedding model choices.

## Deployment

The repository includes a `render.yaml` for deploying to [Render](https://render.com). When deploying, set `GROQ_API_KEY` as an environment variable in the service settings. The first start downloads the embedding model, and vector databases are stored on local disk, so attach a persistent disk if you need indexed data to survive restarts.

## Limitations

- Only text-based PDFs are supported. Scanned documents without a text layer will be reported as empty, since no OCR is performed.
- The app uses a single fixed session ID, so all users of one running instance share the same vector database.
- Answer quality depends on the retrieved context, the embedding model, and the chosen LLM. Adjust the settings in `config.py` to tune behavior.

## Tech Stack

- [Streamlit](https://streamlit.io) for the interface
- [LangChain](https://www.langchain.com) for document loading, splitting, and LLM integration
- [Chroma](https://www.trychroma.com) as the vector database
- [Sentence Transformers](https://www.sbert.net) for embeddings
- [Groq](https://groq.com) for LLM inference
- [pypdf](https://pypdf.readthedocs.io) for PDF parsing

## Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss the approach.

## License

No license has been specified yet. 
