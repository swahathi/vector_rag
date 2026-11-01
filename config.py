
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
    # Streamlit Community Cloud is a 1 GB RAM container; keep these
    # values conservative so the app survives a cold start + ingestion
    # without being OOM-killed.
    max_files: int = 10          # was 1600 — absurd on a 1 GB container
    max_total_mb: int = 50       # was 10000 — would fill /tmp instantly
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_batch_size: int = 100  # was 1000 — embed 1000 chunks = OOM spike

    # Retrieval
    # 20 chunks × ~512 tokens each → very large prompts that can hit
    # Groq's context window hard. 6–8 is sufficient for semantic RAG.
    retrieval_k: int = 6         # was 20

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
