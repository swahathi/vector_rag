import gc
import json
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from config import CONFIG

# One lock per session_id so concurrent Streamlit reruns/threads
# can't open/delete the same on-disk DB at the same time.
_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    with _locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def _session_dir(session_id: str) -> Path:
    return Path(CONFIG.vector_db_root) / session_id


def vector_db_exists(session_id: str) -> bool:
    path = _session_dir(session_id)
    return path.exists() and any(path.iterdir())


def get_db_size_mb(session_id: str) -> float:
    path = _session_dir(session_id)
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def load_manifest(session_id: str) -> dict:
    manifest_path = _session_dir(session_id) / "manifest.json"
    if not manifest_path.exists():
        return {"indexed_files": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(session_id: str, manifest: dict) -> None:
    path = _session_dir(session_id)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 384,
    distance: rest.Distance = rest.Distance.COSINE,
) -> None:
    """Ensure collection exists in Qdrant with the appropriate vector dimension."""
    if not client.collection_exists(collection_name):
        logging.info("Creating local Qdrant collection '%s' (dim=%d)...", collection_name, vector_size)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=distance),
        )


@contextmanager
def vector_db_session(session_id: str, embeddings):
    """
    Opens a Qdrant client scoped to this `with` block only.
    Guarantees the underlying client connection is cleanly closed
    before the block exits.
    """
    lock = _get_session_lock(session_id)
    lock.acquire()

    path = _session_dir(session_id)
    path.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(path))
    collection_name = CONFIG.collection_name
    vector_size = getattr(CONFIG, "embedding_dimension", 384)
    ensure_collection_exists(client, collection_name, vector_size=vector_size)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    try:
        yield store
    finally:
        try:
            client.close()
        except Exception:
            logging.exception("Error while closing Qdrant client")
        finally:
            del store
            del client
            gc.collect()
            lock.release()


def clear_session_db(session_id: str) -> None:
    """
    Deletes the on-disk Qdrant database for a session.
    Safe to call while the app is running because no code path
    stores a client object across Streamlit reruns.
    """
    lock = _get_session_lock(session_id)
    with lock:
        path = _session_dir(session_id)
        if not path.exists():
            return
        gc.collect()
        for attempt in range(3):
            try:
                shutil.rmtree(path)
                return
            except PermissionError as exc:
                if attempt < 2:
                    time.sleep(0.25)
                    gc.collect()
                else:
                    logging.exception("Failed to delete vector DB directory")
                    raise PermissionError(
                        "The database directory could not be deleted. "
                        "If this persists, an external process may be "
                        "holding the file open."
                    ) from exc
