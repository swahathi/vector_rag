import streamlit as st
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings

from config import CONFIG


@st.cache_resource
def load_embeddings():
    base_embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG.embedding_model_name,
        # Force CPU — Streamlit Cloud is CPU-only; without this,
        # sentence-transformers probes for CUDA/MPS on every cold start,
        # wasting ~1–2 s and risking an OOM spike on the 1 GB container.
        model_kwargs={"device": "cpu"},
        # Normalize embeddings so cosine similarity scores are meaningful
        # in ChromaDB (which uses L2 by default; normalized vectors make
        # L2 equivalent to cosine distance).
        encode_kwargs={"normalize_embeddings": True},
    )
    store = LocalFileStore(CONFIG.embed_cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace=CONFIG.embedding_namespace,
        key_encoder="sha256",
    )