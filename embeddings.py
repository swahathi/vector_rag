import os
import streamlit as st
import torch
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings

from config import CONFIG

# ---------------------------------------------------------------------------
# Hugging Face Hub configuration & warning suppression
# ---------------------------------------------------------------------------
# Check if user provided an HF_TOKEN in Streamlit secrets
if "HF_TOKEN" not in os.environ:
    try:
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            os.environ["HF_TOKEN"] = str(st.secrets["HF_TOKEN"])
    except Exception:
        pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import huggingface_hub
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    huggingface_hub.utils.disable_progress_bars()
except Exception:
    pass


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
