import streamlit as st
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings

from config import CONFIG


@st.cache_resource
def load_embeddings():
    base_embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG.embedding_model_name
    )
    store = LocalFileStore(CONFIG.embed_cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace=CONFIG.embedding_namespace,
        key_encoder="sha256",
    )