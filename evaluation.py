from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentSnapshot:
    session_id: str
    notes: str = "Placeholder for future RAGAS, retriever, and embedding evaluations."
