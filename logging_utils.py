import json
import logging
from pathlib import Path
from typing import Any


def configure_logging(log_file: str) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def append_jsonl_record(path: str, payload: dict[str, Any]) -> None:
    record_path = Path(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")