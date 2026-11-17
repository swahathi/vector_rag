# SETUP.md — scalable-ingestion branch
# Complete setup guide for the scalable ingestion architecture with Qdrant Cloud

# Scalable Ingestion — Setup Guide

This branch separates **PDF ingestion** (runs in Google Colab against your Drive) from **querying** (runs in Streamlit, talks to Qdrant Cloud).

---

## Architecture Overview

```
Google Drive (PDFs)
       │
       ▼
 Google Colab
 colab_ingestion.ipynb
   - Loads PDFs from Drive
   - Chunks + embeds locally (GPU-accelerated)
   - Inserts into Qdrant Cloud
   - Saves checkpoint to Drive
       │
       ▼
  Qdrant Cloud
  (managed persistent remote vector store)
       │
       ▼
  Streamlit app_cloud.py
   - Queries Qdrant Cloud
   - Calls Groq for answers
   - No PDF upload needed
```

---

## Step 1 — Get a Qdrant Cloud Account

1. Go to [https://cloud.qdrant.io](https://cloud.qdrant.io) and create an account.
2. Create a new **Cluster** (free 1 GB tier is available).
3. From the dashboard, copy:
   - **Cluster URL** → displayed on your cluster overview, e.g.:
     `https://xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east4-0.gcp.cloud.qdrant.io:6333`
   - **API Key** → Data Access Control → API Keys → Create Key.
   - **Collection Name** → `app_collection` (or your preferred collection name).

---

## Step 2 — Upload PDFs to Google Drive

1. Open [drive.google.com](https://drive.google.com)
2. Create a new folder (e.g. `pdfs`) inside **My Drive**
3. Upload your PDF files into that folder
4. After Drive is mounted in Colab, the path will be:
   `/content/drive/MyDrive/pdfs`

---

## Step 3 — Run the Colab Ingestion Notebook

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Upload `colab_ingestion.ipynb` (File → Upload notebook)
3. In **Cell 3**, fill in:
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `QDRANT_COLLECTION_NAME` (default: `"app_collection"`)
   - The Drive folder path to your PDFs
4. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
5. Run all cells top to bottom (`Runtime → Run all`)
6. When prompted, allow Colab to access your Google Drive
7. The notebook will:
   - Mount your Drive
   - Load the resumable checkpoint (if any)
   - Process each PDF: fingerprint → in-memory load → chunk → embed (GPU) → batch upsert → checkpoint
   - Print progress every 50 files
   - Save a checkpoint to Drive every 10 files

**If Colab disconnects:** Re-run all cells. The checkpoint ensures already-indexed files are skipped automatically.

---

## Step 4 — Configure the Streamlit App

Copy the secrets template:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and fill in:
```toml
QDRANT_URL             = "https://your-cluster-url.cloud.qdrant.io:6333"
QDRANT_API_KEY         = "your-qdrant-api-key"
QDRANT_COLLECTION_NAME = "app_collection"
GROQ_API_KEY           = "your-groq-api-key"
```

---

## Step 5 — Run the Streamlit Query App Locally

Make sure you are on the `scalable-ingestion` branch:
```bash
git checkout scalable-ingestion
```

Install dependencies:
```bash
pip install -r requirements_cloud.txt
```

Run the cloud query app:
```bash
streamlit run app_cloud.py
```

The sidebar will show the Qdrant Cloud connection status and vector count.

---

## Step 6 — Deploy to Streamlit Community Cloud (Optional)

1. Push the `scalable-ingestion` branch to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Repository: your repo, Branch: `scalable-ingestion`, Main file: `app_cloud.py`
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   QDRANT_URL             = "https://your-cluster-url.cloud.qdrant.io:6333"
   QDRANT_API_KEY         = "your-qdrant-api-key"
   QDRANT_COLLECTION_NAME = "app_collection"
   GROQ_API_KEY           = "your-groq-api-key"
   ```
5. Deploy

---

## File Reference

| File | Purpose |
|------|---------|
| `app_cloud.py` | Query-only Streamlit app (replaces `app.py` on this branch) |
| `vectordb_cloud.py` | Qdrant Cloud client (replaces local `vectordb.py`) |
| `config_cloud.py` | Extended config with cloud/Drive settings |
| `checkpoint.py` | Resumable ingestion checkpoint I/O |
| `colab_ingestion.ipynb` | Colab notebook: Drive → Qdrant Cloud ingestion |
| `requirements_cloud.txt` | Dependencies for this branch |
| `.streamlit/secrets.toml.example` | Secrets template (copy → secrets.toml, fill in values) |
| `.streamlit/config.toml` | Streamlit server settings (maxUploadSize, maxMessageSize) |
