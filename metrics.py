from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class IndexingMetrics:
    started_at: float = field(default_factory=time.time)
    pdf_count: int = 0
    dataset_size_mb: float = 0.0
    chunk_size: int = 0
    chunk_overlap: int = 0
    load_time: float = 0.0
    chunk_time: float = 0.0
    embedding_time: float = 0.0
    vector_insertion_time: float = 0.0
    retrieval_latency: float = 0.0
    qdrant_init_time: float = 0.0
    total_indexing_time: float = 0.0
    total_chunks: int = 0
    processed_files: int = 0
    skipped_duplicates: int = 0
    empty_documents: int = 0
    failed_files: int = 0
    db_size_mb: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def chroma_init_time(self) -> float:
        return self.qdrant_init_time

    @chroma_init_time.setter
    def chroma_init_time(self, val: float):
        self.qdrant_init_time = val

    def as_dict(self) -> dict[str, float | int | str | list[str]]:
        return {
            "pdf_count": self.pdf_count,
            "dataset_size_mb": round(self.dataset_size_mb, 2),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "load_time": round(self.load_time, 2),
            "chunk_time": round(self.chunk_time, 2),
            "embedding_time": round(self.embedding_time, 2),
            "vector_insertion_time": round(self.vector_insertion_time, 2),
            "retrieval_latency": round(self.retrieval_latency, 4),
            "qdrant_init_time": round(self.qdrant_init_time, 2),
            "total_indexing_time": round(self.total_indexing_time, 2),
            "total_chunks": self.total_chunks,
            "processed_files": self.processed_files,
            "skipped_duplicates": self.skipped_duplicates,
            "empty_documents": self.empty_documents,
            "failed_files": self.failed_files,
            "db_size_mb": round(self.db_size_mb, 2),
            "errors": self.errors,
        }


@dataclass
class QueryMetrics:
    qdrant_retrieval_time: float = 0.0
    retrieval_time: float = 0.0

    @property
    def chroma_retrieval_time(self) -> float:
        return self.qdrant_retrieval_time

    @chroma_retrieval_time.setter
    def chroma_retrieval_time(self, val: float):
        self.qdrant_retrieval_time = val

    def as_dict(self) -> dict[str, float]:
        return {
            "qdrant_retrieval_time": round(self.qdrant_retrieval_time, 4),
            "retrieval_time": round(self.retrieval_time, 4),
        }
