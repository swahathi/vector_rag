# Multi-PDF Semantic RAG System

A Streamlit-based Retrieval-Augmented Generation (RAG) application for asking natural-language questions across multiple PDF documents.

The system extracts text from uploaded PDFs, splits the documents into overlapping chunks, generates local semantic embeddings using a Sentence Transformer model, stores them in a vector database, retrieves the most relevant chunks for a user query, and passes the retrieved context to a Groq-hosted Llama model to generate a grounded answer.

## Live Demo

The application is deployed using Streamlit Community Cloud and can be accessed from the project deployment page.

## Overview

This project focuses on building and evaluating an end-to-end semantic RAG pipeline for multi-document question answering.

The system provides:

* Multi-PDF document ingestion
* Local semantic embeddings
* Vector similarity search
* Retrieval-augmented generation
* Source-aware answers
* Retrieved passage inspection
* Duplicate document detection
* Batch document processing
* Indexing and retrieval performance metrics
* Structured experiment logging
* Streamlit-based interactive interface

## How It Works

```text
                    PDF Documents
                         |
                         v
                  PDF Text Extraction
                         |
                         v
                  Text Chunking
              (1000 chars, 200 overlap)
                         |
                         v
             Sentence Transformer
              Embedding Generation
                         |
                         v
                    Vector DB
                         |
                         |
User Question ----------+
                         |
                         v
                  Query Embedding
                         |
                         v
              Similarity Search
                    (Top-K = 20)
                         |
                         v
                Retrieved Context
                         |
                         v
                  Groq Llama LLM
                         |
                         v
                Grounded Answer
                         |
                         v
             Sources + Retrieved Chunks
```

### Pipeline

1. **Ingestion**

   Uploaded PDFs are processed and fingerprinted using SHA-256. Duplicate files are detected and skipped.

2. **Text Extraction**

   PDF content is extracted page by page using `PyPDFLoader`.

3. **Chunking**

   Extracted text is divided into overlapping chunks using a recursive character text splitter.

4. **Embedding**

   Each chunk is converted into a semantic vector representation using:

   `sentence-transformers/all-MiniLM-L6-v2`

5. **Vector Storage**

   The generated embeddings and document metadata are stored in the configured vector database.

6. **Retrieval**

   When a user asks a question, the question is embedded and the most semantically relevant document chunks are retrieved using similarity search.

7. **Generation**

   The retrieved chunks are provided as context to the Groq-hosted Llama model, which generates an answer grounded in the retrieved documents.

8. **Presentation**

   The application displays the generated answer, source PDFs, retrieval information, and the retrieved passages used to generate the response.

## Features

### Multi-PDF Support

Upload and process multiple PDF documents within a single session.

### Local Embeddings

Uses `sentence-transformers/all-MiniLM-L6-v2` for local embedding generation, avoiding the need for a separate embedding API.

### Semantic Retrieval

Questions are matched against document chunks using vector similarity rather than simple keyword matching.

### Duplicate Detection

Uploaded files are fingerprinted using SHA-256. If a document has already been indexed, it is skipped instead of being processed again.

### Batch Processing

Documents are processed in batches with individual file status reporting, including:

* Indexed
* Duplicate
* Empty
* Failed

### Source-Aware Answers

The application identifies the PDF documents contributing to the retrieved context and allows users to inspect the retrieved passages.

### Retrieval Transparency

Retrieved chunks expose metadata such as:

* Source PDF
* Page number
* Chunk ID
* Chunk index
* Similarity score
* Character count

### Experiment Metrics

The application records timing and size metrics for indexing and querying, making it possible to experiment with different RAG configurations.

## Project Structure

| File               | Responsibility                                                |
| ------------------ | ------------------------------------------------------------- |
| `app.py`           | Streamlit interface and application orchestration             |
| `config.py`        | Central application configuration                             |
| `ingestion.py`     | PDF loading, deduplication, metadata tagging and indexing     |
| `chunking.py`      | Text splitter configuration                                   |
| `embeddings.py`    | Embedding model loading                                       |
| `vectordb.py`      | Vector database lifecycle and storage                         |
| `retrieval.py`     | Similarity search, context construction and prompt generation |
| `llm.py`           | Groq LLM configuration                                        |
| `metrics.py`       | Indexing and query metric structures                          |
| `evaluation.py`    | Experiment and evaluation support                             |
| `logging_utils.py` | Application and experiment logging                            |
| `render.yaml`      | Deployment configuration, if applicable                       |
| `requirements.txt` | Python dependencies                                           |
| `README.md`        | Project documentation                                         |

## Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Sentence Transformers**
* **Vector Database**
* **Groq**
* **Llama**
* **PyPDF**
* **Pandas**
* **NumPy**

## Requirements

* Python 3.10+
* Groq API key
* Sufficient storage for the embedding model and vector database
* Internet connection for downloading required Python packages and the embedding model

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

## Configuration

The application requires a Groq API key for question answering.

For local execution, configure:

```bash
GROQ_API_KEY="your-api-key"
```

For Streamlit Community Cloud deployment, configure `GROQ_API_KEY` through the application's Streamlit Secrets settings.

The API key should never be committed to GitHub.

## Usage

Run the application locally:

```bash
streamlit run app.py
```

Then:

1. Upload one or more PDF documents.
2. Click **Process PDFs**.
3. Wait for the indexing process to complete.
4. Enter a natural-language question.
5. Click **Get Answer**.
6. Review the generated answer and source documents.
7. Expand the retrieved chunks to inspect the context used by the RAG pipeline.

Additional PDFs can be uploaded and indexed during the session.

## Metrics and Experimentation

The application records metrics for each indexing and retrieval operation.

### Indexing Metrics

The system tracks:

* PDF loading time
* Chunking time
* Embedding time
* Vector insertion time
* Total processing time
* Vector database size
* Number of PDFs
* Number of chunks
* Dataset size
* Chunk size
* Chunk overlap

### Query Metrics

The system records:

* Vector retrieval time
* Overall retrieval time
* Number of retrieved chunks
* Configured retrieval `k`

These metrics make the application useful for experimenting with different:

* Chunk sizes
* Chunk overlaps
* Retrieval values
* Embedding models
* Vector database configurations

## Experiment Logging

The application maintains structured logs for pipeline execution and experiments.

`rag.log` records application events, processing steps, failures, and queries.

`experiment_runs.jsonl` stores structured experiment records that can be analysed using Python, Pandas, or spreadsheet software.

## Limitations

* Only text-based PDFs are supported.
* Scanned PDFs without an available text layer require OCR, which is not currently implemented.
* Retrieval quality depends on the embedding model, chunking strategy and retrieval configuration.
* Answer quality depends on the retrieved context and the selected LLM.
* Streamlit deployment environments have resource and execution limitations for very large document collections.
* API-based LLM generation requires a valid Groq API key.

## Future Improvements

Potential extensions include:

* Hybrid keyword + semantic retrieval
* Reranking retrieved chunks
* Improved evaluation using RAG-specific metrics
* Support for OCR-based PDFs
* Metadata filtering
* Multiple embedding model comparisons
* Retrieval latency benchmarking
* Larger-scale vector database evaluation
* Automated question-answer evaluation
* Improved multi-user session isolation

## Why This Project

This project was developed to explore the practical implementation and evaluation of Retrieval-Augmented Generation systems over multiple PDF documents.

The main focus is not only generating answers, but also understanding the individual stages of a RAG pipeline, including document ingestion, chunking, embedding generation, vector indexing, similarity retrieval, context construction, LLM generation, and performance measurement.

## Deployment

The application is deployed using **Streamlit Community Cloud**.

The deployed application requires the `GROQ_API_KEY` to be configured through Streamlit Secrets for question answering to work.

## Contributing

Issues and pull requests are welcome.

For significant changes, open an issue first to discuss the proposed approach.

## License

No license has been specified yet.

If this project is intended to be distributed publicly, add an appropriate `LICENSE` file such as MIT or Apache-2.0.
