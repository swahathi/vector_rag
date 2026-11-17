"""
vectordb_cloud.py
-----------------
Qdrant Cloud client for production vector search.

Connects to Qdrant Cloud (or any remote Qdrant instance) via qdrant-client.
The Streamlit query app and ingestion notebook both share the same collection.

Credentials are read from:
  - Streamlit Secrets (st.secrets) when running inside Streamlit
  - Environment variables when running in Colab or locally

Required secrets / env vars:
    QDRANT_URL             – Qdrant Cloud cluster endpoint (e.g. https://xxx.cloud.qdrant.io:6333)
    QDRANT_API_KEY         – Qdrant Cloud API key
    QDRANT_COLLECTION_NAME – Collection name (defaults to "app_collection")
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from config import CONFIG


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str | None = None) -> str:
    """
    Read a credential from Streamlit secrets if available, otherwise
    fall back to an environment variable. Raises RuntimeError if the
    required value is missing in both places.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            value = st.secrets.get(key)
            qdrant_section = st.secrets.get("qdrant")
            if not value and isinstance(qdrant_section, dict):
                value = qdrant_section.get(key)
            if value:
                return str(value)
    except Exception:
        pass

    value = os.getenv(key)
    if value:
        return value
    if default is not None:
        return default
    raise RuntimeError(
        f"Missing required credential: '{key}'. "
        "Set it in .streamlit/secrets.toml (local), "
        "Streamlit Cloud Secrets (deployed), or as an environment variable."
    )


def get_qdrant_credentials() -> tuple[str, str | None, str]:
    """Return (url, api_key, collection_name) from secrets / env vars."""
    url = _get_secret("QDRANT_URL")
    api_key = _get_secret("QDRANT_API_KEY", default="")
    collection_name = _get_secret(
        "QDRANT_COLLECTION_NAME",
        default=getattr(CONFIG, "collection_name", "app_collection"),
    )
    return url, (api_key or None), collection_name


# ---------------------------------------------------------------------------
# Client factory & collection manager
# ---------------------------------------------------------------------------

def make_cloud_client() -> QdrantClient:
    """Create a QdrantClient using stored credentials."""
    url, api_key, _ = get_qdrant_credentials()
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 384,
    distance: rest.Distance = rest.Distance.COSINE,
) -> None:
    """Ensure the target collection exists in Qdrant with the appropriate vector dimension."""
    if not client.collection_exists(collection_name):
        logging.info("Creating Qdrant collection '%s' (dim=%d, distance=%s)...", collection_name, vector_size, distance)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=distance),
        )


# ---------------------------------------------------------------------------
# Public API (mirrors the interface used throughout the app)
# ---------------------------------------------------------------------------

@contextmanager
def vector_db_session(embeddings):
    """
    Context manager that yields a LangChain QdrantVectorStore backed by
    Qdrant Cloud.
    """
    url, api_key, collection_name = get_qdrant_credentials()
    client = QdrantClient(url=url, api_key=api_key)
    vector_size = getattr(CONFIG, "embedding_dimension", 384)
    ensure_collection_exists(client, collection_name, vector_size=vector_size)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    try:
        yield store
    except Exception:
        logging.exception("Error during Qdrant Cloud vector_db_session")
        raise
    finally:
        client.close()


def collection_count() -> int:
    """Return the number of vectors currently stored in the cloud collection."""
    try:
        client = make_cloud_client()
        _, _, collection_name = get_qdrant_credentials()
        if not client.collection_exists(collection_name):
            client.close()
            return 0
        count = client.count(collection_name=collection_name).count
        client.close()
        return count
    except Exception:
        return 0


def cloud_collection_exists() -> bool:
    """Return True if the cloud collection has been created and contains data."""
    return collection_count() > 0
