"""
checkpoint.py
-------------
Per-file resumable ingestion checkpoint for the scalable-ingestion branch.

The checkpoint is a JSON file saved to Google Drive (or any writable path).
It records the SHA-256 fingerprint of every successfully indexed PDF so
the ingestion notebook can skip already-processed files if it is interrupted
and restarted.

The JSON format mirrors the manifest used in the local vectordb.py so the
two deduplication mechanisms are consistent:

{
    "indexed_files": {
        "<sha256-fingerprint>": {
            "file_name": "report.pdf",
            "size_bytes": 123456,
            "chunk_count": 42,
            "indexed_at": 1700000000.0
        },
        ...
    },
    "runs": [
        { "timestamp": ..., "files_indexed": ..., "total_chunks": ... },
        ...
    ]
}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Core checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(path: str | Path) -> dict:
    """
    Load the checkpoint JSON from *path*.
    Returns an empty checkpoint dict if the file does not exist yet.
    """
    p = Path(path)
    if not p.exists():
        logging.info("No checkpoint found at %s — starting fresh.", p)
        return {"indexed_files": {}, "runs": []}
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    count = len(data.get("indexed_files", {}))
    logging.info("Checkpoint loaded from %s — %d files already indexed.", p, count)
    return data


def save_checkpoint(path: str | Path, checkpoint: dict) -> None:
    """
    Atomically write *checkpoint* to *path* (write to a temp file then rename).
    Creates parent directories if necessary.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, indent=2)
    tmp.replace(p)  # Atomic on POSIX; near-atomic on Windows


# ---------------------------------------------------------------------------
# High-level helpers used by the Colab ingestion notebook
# ---------------------------------------------------------------------------

def is_already_indexed(checkpoint: dict, fingerprint: str) -> bool:
    """Return True if this SHA-256 fingerprint is recorded in the checkpoint."""
    return fingerprint in checkpoint.get("indexed_files", {})


def record_indexed_file(
    checkpoint: dict,
    fingerprint: str,
    file_name: str,
    size_bytes: int,
    chunk_count: int,
) -> None:
    """
    Add a successfully indexed file to the in-memory checkpoint.
    Call save_checkpoint() afterwards to persist to disk.
    """
    checkpoint.setdefault("indexed_files", {})[fingerprint] = {
        "file_name": file_name,
        "size_bytes": size_bytes,
        "chunk_count": chunk_count,
        "indexed_at": time.time(),
    }


def append_run_summary(
    checkpoint: dict,
    files_indexed: int,
    files_skipped: int,
    files_failed: int,
    total_chunks: int,
) -> None:
    """Append a summary record for the current ingestion run."""
    checkpoint.setdefault("runs", []).append(
        {
            "timestamp": time.time(),
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "files_failed": files_failed,
            "total_chunks": total_chunks,
        }
    )


def checkpoint_summary(checkpoint: dict) -> str:
    """Return a human-readable one-line summary of the checkpoint state."""
    indexed = len(checkpoint.get("indexed_files", {}))
    runs = len(checkpoint.get("runs", []))
    return f"{indexed} files indexed across {runs} previous run(s)"
