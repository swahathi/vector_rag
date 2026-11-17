import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Multi PDF RAG System"
    page_layout: str = "wide"
    app_title: str = "Multi-PDF Semantic RAG System"

    # Storage
    embed_cache_dir: str = "./embedding_cache"
    vector_db_root: str = "./qdrant_sessions"
    log_file: str = "./rag.log"
    experiment_log_file: str = "./experiment_runs.jsonl"
    llm_cache_file: str = "./langchain_cache.db"

    # Vector DB / Qdrant settings
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "app_collection")
    embedding_dimension: int = 384

    # Document processing
    max_files: int = 1600
    max_total_mb: int = 200
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_batch_size: int = 100

    # Retrieval
    retrieval_k: int = 6

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_namespace: str = "all-MiniLM-L6-v2"

    # LLM
    groq_model_name: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    @property
    def vector_db_root_path(self) -> Path:
        return Path(self.vector_db_root)


CONFIG = AppConfig()
