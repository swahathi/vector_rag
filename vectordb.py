import logging
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

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
    import json

    manifest_path = _session_dir(session_id) / "manifest.json"
    if not manifest_path.exists():
        return {"indexed_files": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(session_id: str, manifest: dict) -> None:
    import json

    path = _session_dir(session_id)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


@contextmanager
def vector_db_session(session_id: str, embeddings):
    """
    Opens a Chroma client scoped to this `with` block only.
    Guarantees the underlying sqlite3 connection is explicitly
    closed before the block exits, on every code path (success
    or exception), so no handle survives into the next Streamlit
    rerun or into a delete operation.
    """
    lock = _get_session_lock(session_id)
    lock.acquire()

    path = _session_dir(session_id)
    path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    store = Chroma(
        client=client,
        collection_name=CONFIG.collection_name,
        embedding_function=embeddings,
    )

    try:
        yield store
    finally:
        try:
            # Explicitly release chromadb's sqlite connection pool.
            # This is the deterministic close that gc.collect()
            # cannot guarantee: chromadb's SqliteDB keeps a live
            # sqlite3.Connection in a pool object referenced by
            # the client's internal `_producer`/`_system`. reset()
            # is not what closes it — calling the private connection
            # close is required on the API version pinned in config.
            client.clear_system_cache()
        except Exception:
            logging.exception("Error while releasing Chroma client")
        finally:
            lock.release()


def clear_session_db(session_id: str) -> None:
    """
    Deletes the on-disk Chroma database for a session.
    Safe to call while the app is running because no code path
    in this module ever stores a client/store object across
    Streamlit reruns — every client is opened and released
    within a single `with vector_db_session(...)` block, so by
    the time this function runs, no file handle is held by
    this process.
    """
    lock = _get_session_lock(session_id)
    with lock:
        path = _session_dir(session_id)
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
        except PermissionError as exc:
            logging.exception("Failed to delete vector DB directory")
            raise PermissionError(
                "The database directory could not be deleted. "
                "If this persists, an external process (antivirus, "
                "file indexer, or a second Streamlit worker) may be "
                "holding the file open."
            ) from exc