from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Multi PDF RAG System"
    page_layout: str = "wide"
    app_title: str = "Multi-PDF Semantic RAG System"

    embed_cache_dir: str = "./embedding_cache"
    vector_db_root: str = "./vector_db"
    log_file: str = "./rag.log"
    experiment_log_file: str = "./experiment_runs.jsonl"
    llm_cache_file: str = "./langchain_cache.db"
    vector_db_root: str = "./chroma_sessions"
    collection_name: str = "rag_collection"

    max_files: int = 1600
    max_total_mb: int = 10000
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_batch_size: int = 1000
    retrieval_k: int = 20

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_namespace: str = "all-MiniLM-L6-v2"
    groq_model_name: str = "llama-3.1-8b-instant"

    @property
    def vector_db_root_path(self) -> Path:
        return Path(self.vector_db_root)


CONFIG = AppConfig()
