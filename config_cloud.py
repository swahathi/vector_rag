"""
config_cloud.py
---------------
Configuration for the scalable-ingestion branch with Qdrant Cloud.

Extends the base AppConfig with cloud-specific settings while keeping
all existing values unchanged. Import CONFIG_CLOUD in cloud-aware
modules instead of CONFIG.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from config import AppConfig, CONFIG


@dataclass(frozen=True)
class CloudAppConfig(AppConfig):
    # -----------------------------------------------------------------------
    # Qdrant Cloud
    # -----------------------------------------------------------------------
    # Collection is shared across ingestion (Colab) and querying (Streamlit).
    # Default collection name: "app_collection"
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "app_collection")

    # -----------------------------------------------------------------------
    # Google Drive paths (set before running the Colab notebook)
    # -----------------------------------------------------------------------
    drive_pdf_folder: str = "MyDrive/pdfs"
    drive_checkpoint_path: str = "MyDrive/rag_checkpoint/ingestion_checkpoint.json"

    # -----------------------------------------------------------------------
    # Ingestion tuning
    # -----------------------------------------------------------------------
    cloud_insert_batch_size: int = 100
    progress_report_interval: int = 50


# Singleton used throughout the scalable-ingestion branch
CONFIG_CLOUD = CloudAppConfig()
