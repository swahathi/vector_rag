from __future__ import annotations
import traceback
import gc
import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from config import CONFIG
from metrics import IndexingMetrics
from vectordb import get_db_size_mb, load_manifest, save_manifest


@dataclass
class FileIngestionResult:
    file_name: str
    status: str
    chunk_count: int = 0
    error: str | None = None


@dataclass
class IngestionSummary:
    total_chunks: int = 0
    file_results: list[FileIngestionResult] = field(default_factory=list)


def fingerprint_uploaded_file(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()


def _save_uploaded_file(uploaded_file, session_id: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    temp_dir = Path(tempfile.gettempdir()) / "streamlit_rag_uploads" / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{hashlib.md5(uploaded_file.name.encode('utf-8')).hexdigest()}{suffix}"
    with temp_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return temp_path


def _add_documents_in_batches(vector_db, chunks, batch_size: int) -> tuple[float, float]:
    embedding_time = 0.0
    insertion_time = 0.0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_start = time.time()
        vector_db.add_documents(batch)
        elapsed = time.time() - batch_start
        embedding_time += elapsed
        insertion_time += elapsed
    return embedding_time, insertion_time


def ingest_uploaded_files(
    uploaded_files,
    session_id: str,
    vector_db,
    splitter,
) -> tuple[IndexingMetrics, IngestionSummary]:
    metrics = IndexingMetrics(
        pdf_count=len(uploaded_files),
        dataset_size_mb=sum(file.size for file in uploaded_files) / (1024 * 1024),
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    summary = IngestionSummary()
    manifest = load_manifest(session_id)
    indexed_files = manifest.setdefault("indexed_files", {})

    for uploaded_file in uploaded_files:
        fingerprint = fingerprint_uploaded_file(uploaded_file)
        if fingerprint in indexed_files:
            metrics.skipped_duplicates += 1
            summary.file_results.append(
                FileIngestionResult(uploaded_file.name, "duplicate")
            )
            logging.info("Skipping duplicate upload: %s", uploaded_file.name)
            continue

        temp_path = None
        try:
            load_start = time.time()
            temp_path = _save_uploaded_file(uploaded_file, session_id)
            docs = PyPDFLoader(str(temp_path)).load()
            metrics.load_time += time.time() - load_start

            if not docs:
                metrics.empty_documents += 1
                summary.file_results.append(
                    FileIngestionResult(uploaded_file.name, "empty")
                )
                logging.warning("Empty PDF: %s", uploaded_file.name)
                continue

            for doc in docs:
                doc.metadata["source_pdf"] = uploaded_file.name
                doc.metadata["session_id"] = session_id
                doc.metadata["file_fingerprint"] = fingerprint

            chunk_start = time.time()
            chunks = splitter.split_documents(docs)

            if not chunks:
                metrics.empty_documents+=1
                summary.file_results.append(
                    FileIngestionResult(uploaded_file.name,"empty")
                )
                logging.warning("PDF produced no chunks: %s",uploaded_file.name)
                continue
            total_chunks=len(chunks)
            for idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "filename": uploaded_file.name,
                    "session_id": session_id,
                    "file_fingerprint": fingerprint,
                    "chunk_id": f"{fingerprint}_{idx}",
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "page": chunk.metadata.get("page", -1),
                    "char_count": len(chunk.page_content),
    })
            metrics.chunk_time += time.time() - chunk_start

            if not chunks:
                metrics.empty_documents += 1
                summary.file_results.append(
                    FileIngestionResult(uploaded_file.name, "empty")
                )
                logging.warning("PDF produced no chunks: %s", uploaded_file.name)
                continue

            embed_time, insert_time = _add_documents_in_batches(
                vector_db,
                chunks,
                CONFIG.chunk_batch_size,
            )
            metrics.embedding_time += embed_time
            metrics.vector_insertion_time += insert_time
            metrics.total_chunks += len(chunks)
            metrics.processed_files += 1

            indexed_files[fingerprint] = {
                "file_name": uploaded_file.name,
                "size_bytes": uploaded_file.size,
                "chunk_count": len(chunks),
                "indexed_at": time.time(),
            }
            save_manifest(session_id, manifest)
            summary.total_chunks += len(chunks)
            summary.file_results.append(
                FileIngestionResult(
                    file_name=uploaded_file.name,
                    status="indexed",
                    chunk_count=len(chunks),
                )
            )
            logging.info(
                "Indexed %s with %s chunks",
                uploaded_file.name,
                len(chunks),
            )
        except Exception as exc:
            metrics.failed_files += 1
            error_message = f"{uploaded_file.name}: {exc}"
            metrics.errors.append(error_message)
            print("\n" + "=" * 80)
            print(f"FAILED PDF : {uploaded_file.name}")
            print(f"ERROR TYPE : {type(exc).__name__}")
            print(f"ERROR      : {exc}")
            traceback.print_exc()
            logging.exception("Failed to ingest %s", uploaded_file.name)
            print("=" * 80 + "\n")
            summary.file_results.append(
                FileIngestionResult(
                    file_name=uploaded_file.name,
                    status="failed",
                    error=str(exc),
                )
            )
            logging.exception("Failed to ingest %s", uploaded_file.name)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            gc.collect()

    metrics.total_indexing_time = time.time() - metrics.started_at
    metrics.db_size_mb = get_db_size_mb(session_id)
    manifest.setdefault("runs", []).append(
        {
            "timestamp": time.time(),
            "metrics": metrics.as_dict(),
        }
    )
    save_manifest(session_id, manifest)
    return metrics, summary
