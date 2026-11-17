from __future__ import annotations

import gc
import hashlib
import io
import logging
import os
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from qdrant_client.http import models as rest

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


@dataclass
class _FileParsedData:
    uploaded_file: Any
    file_name: str
    file_size: int
    fingerprint: str
    chunks: list[Document]
    load_time: float
    chunk_time: float
    error: str | None = None
    is_duplicate: bool = False
    is_empty: bool = False


def fingerprint_uploaded_file(uploaded_file) -> str:
    """Compute SHA-256 fingerprint from uploaded file buffer or path."""
    if hasattr(uploaded_file, "getvalue"):
        return hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if hasattr(uploaded_file, "getbuffer"):
        return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
    if isinstance(uploaded_file, (str, Path)):
        return hashlib.sha256(Path(uploaded_file).read_bytes()).hexdigest()
    raise TypeError(f"Unsupported file type: {type(uploaded_file)}")


def _get_file_bytes(file_obj) -> bytes:
    """Retrieve raw bytes from file object or path."""
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    if hasattr(file_obj, "getbuffer"):
        return bytes(file_obj.getbuffer())
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj).read_bytes()
    raise TypeError(f"Cannot read bytes from: {type(file_obj)}")


def extract_pdf_documents(file_obj) -> list[Document]:
    """
    Fast in-memory PDF text extraction using pypdf.
    Falls back to temporary disk file with PyPDFLoader if in-memory parsing fails.
    """
    file_name = getattr(file_obj, "name", str(file_obj))
    file_bytes = _get_file_bytes(file_obj)

    # 1. Fast in-memory parsing
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        docs: list[Document] = []
        for page_idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"page": page_idx + 1},
                        )
                    )
            except Exception as page_err:
                logging.warning("Error reading page %d of %s: %s", page_idx, file_name, page_err)

        if docs:
            return docs
    except Exception as read_err:
        logging.warning("Direct in-memory PDF parse failed for %s: %s. Using PyPDFLoader fallback.", file_name, read_err)

    # 2. Fallback to PyPDFLoader with temp file
    temp_path = None
    try:
        suffix = Path(file_name).suffix or ".pdf"
        temp_dir = Path(tempfile.gettempdir()) / "streamlit_rag_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{hashlib.md5(file_name.encode('utf-8')).hexdigest()}_{time.time()}{suffix}"
        temp_path.write_bytes(file_bytes)
        return PyPDFLoader(str(temp_path)).load()
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def prepare_chunks(
    docs: list[Document],
    file_name: str,
    fingerprint: str,
    session_id: str,
    splitter,
) -> list[Document]:
    """Split documents into chunks and attach complete metadata."""
    for doc in docs:
        doc.metadata["source_pdf"] = file_name
        doc.metadata["filename"] = file_name
        doc.metadata["session_id"] = session_id
        doc.metadata["file_fingerprint"] = fingerprint

    chunks = splitter.split_documents(docs)
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "filename": file_name,
                "source_pdf": file_name,
                "session_id": session_id,
                "file_fingerprint": fingerprint,
                "chunk_id": f"{fingerprint}_{idx}",
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "page": chunk.metadata.get("page", -1),
                "char_count": len(chunk.page_content),
            }
        )
    return chunks


def _parse_file(uploaded_file, session_id: str, indexed_files: dict, splitter) -> _FileParsedData:
    file_name = getattr(uploaded_file, "name", str(uploaded_file))
    file_size = getattr(uploaded_file, "size", 0)
    try:
        fingerprint = fingerprint_uploaded_file(uploaded_file)
    except Exception as exc:
        return _FileParsedData(
            uploaded_file=uploaded_file,
            file_name=file_name,
            file_size=file_size,
            fingerprint="",
            chunks=[],
            load_time=0.0,
            chunk_time=0.0,
            error=str(exc),
        )

    if fingerprint in indexed_files:
        return _FileParsedData(
            uploaded_file=uploaded_file,
            file_name=file_name,
            file_size=file_size,
            fingerprint=fingerprint,
            chunks=[],
            load_time=0.0,
            chunk_time=0.0,
            is_duplicate=True,
        )

    load_start = time.time()
    try:
        docs = extract_pdf_documents(uploaded_file)
    except Exception as exc:
        return _FileParsedData(
            uploaded_file=uploaded_file,
            file_name=file_name,
            file_size=file_size,
            fingerprint=fingerprint,
            chunks=[],
            load_time=time.time() - load_start,
            chunk_time=0.0,
            error=str(exc),
        )
    load_time = time.time() - load_start

    if not docs:
        return _FileParsedData(
            uploaded_file=uploaded_file,
            file_name=file_name,
            file_size=file_size,
            fingerprint=fingerprint,
            chunks=[],
            load_time=load_time,
            chunk_time=0.0,
            is_empty=True,
        )

    chunk_start = time.time()
    try:
        chunks = prepare_chunks(docs, file_name, fingerprint, session_id, splitter)
    except Exception as exc:
        return _FileParsedData(
            uploaded_file=uploaded_file,
            file_name=file_name,
            file_size=file_size,
            fingerprint=fingerprint,
            chunks=[],
            load_time=load_time,
            chunk_time=time.time() - chunk_start,
            error=str(exc),
        )
    chunk_time = time.time() - chunk_start

    if not chunks:
        return _FileParsedData(
            uploaded_file=uploaded_file,
            file_name=file_name,
            file_size=file_size,
            fingerprint=fingerprint,
            chunks=[],
            load_time=load_time,
            chunk_time=chunk_time,
            is_empty=True,
        )

    return _FileParsedData(
        uploaded_file=uploaded_file,
        file_name=file_name,
        file_size=file_size,
        fingerprint=fingerprint,
        chunks=chunks,
        load_time=load_time,
        chunk_time=chunk_time,
    )


def _add_documents_in_batches(vector_db, chunks: list[Document], batch_size: int) -> tuple[float, float]:
    """
    Insert chunks into vector_db in batches.
    Separates embedding time from insertion time accurately.
    Works with QdrantVectorStore and other LangChain vector stores.
    """
    embedding_time = 0.0
    insertion_time = 0.0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [doc.page_content for doc in batch]

        # Generate deterministic UUIDs from chunk_id for Qdrant compatibility
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata.get("chunk_id", f"{start}_{i}")))
            for i, doc in enumerate(batch)
        ]

        # Check if vector_db is a QdrantVectorStore with direct client access
        if (
            hasattr(vector_db, "client")
            and hasattr(vector_db, "collection_name")
            and hasattr(vector_db, "embeddings")
            and vector_db.embeddings
        ):
            embed_start = time.time()
            embeddings_list = vector_db.embeddings.embed_documents(texts)
            embedding_time += time.time() - embed_start

            insert_start = time.time()
            points = [
                rest.PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
                for pid, vec, doc in zip(point_ids, embeddings_list, batch)
            ]
            vector_db.client.upsert(
                collection_name=vector_db.collection_name,
                points=points,
            )
            insertion_time += time.time() - insert_start
        else:
            batch_start = time.time()
            vector_db.add_documents(batch, ids=point_ids)
            elapsed = time.time() - batch_start
            embedding_time += elapsed / 2.0
            insertion_time += elapsed / 2.0

    return embedding_time, insertion_time


def ingest_uploaded_files(
    uploaded_files,
    session_id: str,
    vector_db,
    splitter,
) -> tuple[IndexingMetrics, IngestionSummary]:
    """
    High-performance PDF ingestion pipeline:
    - Multi-threaded text extraction and chunking in memory (no disk overhead)
    - Batched embedding and Qdrant vector insertion
    - Accurate metrics and resumable deduplication manifest
    """
    metrics = IndexingMetrics(
        pdf_count=len(uploaded_files),
        dataset_size_mb=sum(getattr(f, "size", 0) for f in uploaded_files) / (1024 * 1024),
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    summary = IngestionSummary()
    manifest = load_manifest(session_id)
    indexed_files = manifest.setdefault("indexed_files", {})

    # 1. Parallel in-memory PDF extraction and chunking
    max_workers = min(8, max(1, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        parsed_results = list(
            executor.map(
                lambda f: _parse_file(f, session_id, indexed_files, splitter),
                uploaded_files,
            )
        )

    all_new_chunks: list[Document] = []
    successful_files_info: list[tuple[str, str, int, int]] = []  # (fingerprint, name, size, chunk_count)

    for parsed in parsed_results:
        metrics.load_time += parsed.load_time
        metrics.chunk_time += parsed.chunk_time

        if parsed.is_duplicate:
            metrics.skipped_duplicates += 1
            summary.file_results.append(FileIngestionResult(parsed.file_name, "duplicate"))
            logging.info("Skipping duplicate upload: %s", parsed.file_name)
            continue

        if parsed.error:
            metrics.failed_files += 1
            error_message = f"{parsed.file_name}: {parsed.error}"
            metrics.errors.append(error_message)
            logging.exception("Failed to ingest %s: %s", parsed.file_name, parsed.error)
            summary.file_results.append(
                FileIngestionResult(
                    file_name=parsed.file_name,
                    status="failed",
                    error=parsed.error,
                )
            )
            continue

        if parsed.is_empty:
            metrics.empty_documents += 1
            summary.file_results.append(FileIngestionResult(parsed.file_name, "empty"))
            logging.warning("Empty or unextractable PDF: %s", parsed.file_name)
            continue

        # File succeeded
        all_new_chunks.extend(parsed.chunks)
        successful_files_info.append(
            (parsed.fingerprint, parsed.file_name, parsed.file_size, len(parsed.chunks))
        )
        summary.file_results.append(
            FileIngestionResult(
                file_name=parsed.file_name,
                status="indexed",
                chunk_count=len(parsed.chunks),
            )
        )

    # 2. Batched vector embedding and insertion
    if all_new_chunks:
        embed_time, insert_time = _add_documents_in_batches(
            vector_db,
            all_new_chunks,
            CONFIG.chunk_batch_size,
        )
        metrics.embedding_time += embed_time
        metrics.vector_insertion_time += insert_time
        metrics.total_chunks += len(all_new_chunks)
        metrics.processed_files += len(successful_files_info)
        summary.total_chunks += len(all_new_chunks)

        # 3. Update manifest
        now = time.time()
        for fp, name, size, count in successful_files_info:
            indexed_files[fp] = {
                "file_name": name,
                "size_bytes": size,
                "chunk_count": count,
                "indexed_at": now,
            }
        save_manifest(session_id, manifest)
        logging.info("Successfully indexed %d chunks across %d files", len(all_new_chunks), len(successful_files_info))

    gc.collect()

    # 4. Finalize metrics
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
