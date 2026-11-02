
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Multi PDF RAG System"
    page_layout: str = "wide"
    app_title: str = "Multi-PDF Semantic RAG System"

    # Storage
    embed_cache_dir: str = "./embedding_cache"
    vector_db_root: str = "./chroma_sessions"
    log_file: str = "./rag.log"
    experiment_log_file: str = "./experiment_runs.jsonl"
    llm_cache_file: str = "./langchain_cache.db"

    collection_name: str = "rag_collection"

    # Document processing
    # max_files / max_total_mb are upload validation guards only - PDFs
    # are processed sequentially and temp files deleted after each one,
    # so these limits do not drive peak RAM usage.
    max_files: int = 1600
    max_total_mb: int = 10000
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Keep batch size modest: limits per-batch embedding RAM spike
    chunk_batch_size: int = 100

    # Retrieval
    retrieval_k: int = 6

    # Embeddings
    embedding_model_name: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_namespace: str = "all-MiniLM-L6-v2"

    # LLM
    groq_model_name: str = "llama-3.1-8b-instant"

    @property
    def vector_db_root_path(self) -> Path:
        return Path(self.vector_db_root)


CONFIG = AppConfig()
