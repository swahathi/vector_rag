from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma

from config import CONFIG


def get_session_db_dir(session_id: str) -> Path:
    return Path(CONFIG.vector_db_root) / session_id


def get_manifest_path(session_id: str) -> Path:
    return get_session_db_dir(session_id) / "session_manifest.json"


def open_vector_db(session_id: str, embeddings) -> Chroma:
    session_dir = get_session_db_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(session_dir),
        embedding_function=embeddings,
    )


def clear_session_db(session_id: str) -> None:
    session_dir = get_session_db_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)


def get_db_size_mb(session_id: str) -> float:
    session_dir = get_session_db_dir(session_id)
    if not session_dir.exists():
        return 0.0

    total_bytes = sum(
        file_path.stat().st_size
        for file_path in session_dir.rglob("*")
        if file_path.is_file()
    )
    return total_bytes / (1024 * 1024)


def load_manifest(session_id: str) -> dict[str, Any]:
    manifest_path = get_manifest_path(session_id)
    if not manifest_path.exists():
        return {"indexed_files": {}, "runs": []}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(session_id: str, manifest: dict[str, Any]) -> None:
    manifest_path = get_manifest_path(session_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def vector_db_exists(session_id: str) -> bool:
    return get_session_db_dir(session_id).exists()
