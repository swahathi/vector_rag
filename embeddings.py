import streamlit as st
import torch
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings

from config import CONFIG


def get_embedding_device() -> str:
    """Detect available hardware accelerator safely."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@st.cache_resource
def load_embeddings():
    device = get_embedding_device()
    base_embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG.embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    store = LocalFileStore(CONFIG.embed_cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace=CONFIG.embedding_namespace,
        key_encoder="sha256",
    )
