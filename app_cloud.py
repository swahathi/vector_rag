"""
app_cloud.py
------------
Streamlit query-only application for the scalable-ingestion branch with Qdrant Cloud.

This app connects to Qdrant Cloud for retrieval and Groq for generation.
It does NOT handle PDF ingestion — all ingestion is done in the Colab
notebook (colab_ingestion.ipynb) and stored in the shared Qdrant Cloud
collection.
"""

import logging
import os
import time

import streamlit as st

from config import CONFIG
from embeddings import load_embeddings
from llm import get_groq_api_key, load_llm
from logging_utils import configure_logging
from retrieval import build_context, build_prompt, retrieve_documents
from vectordb_cloud import cloud_collection_exists, collection_count, vector_db_session

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title=CONFIG.page_title, layout=CONFIG.page_layout)
st.title(CONFIG.app_title)
st.caption("Query-only mode — PDFs are indexed in Qdrant Cloud via the Colab ingestion notebook.")
configure_logging(CONFIG.log_file)

if "pipeline_failed" not in st.session_state:
    st.session_state.pipeline_failed = False

# ---------------------------------------------------------------------------
# Load models (cached across reruns)
# ---------------------------------------------------------------------------

embeddings = load_embeddings()
llm = load_llm()

# ---------------------------------------------------------------------------
# Sidebar — cloud collection status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Qdrant Cloud")
    try:
        count = collection_count()
        st.success(f"✅ Connected — {count:,} vectors indexed")
    except Exception as exc:
        st.error(f"❌ Cannot connect to Qdrant Cloud: {exc}")
        st.info(
            "Check that QDRANT_URL and QDRANT_API_KEY "
            "are set in `.streamlit/secrets.toml` (local) or "
            "Streamlit Cloud Secrets (deployed)."
        )

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Ingestion runs in Google Colab against your Drive PDFs. "
        "This app only queries the indexed Qdrant collection."
    )


# ---------------------------------------------------------------------------
# Helper — reusable step runner
# ---------------------------------------------------------------------------

def run_step(label, func, *args, **kwargs):
    with st.status(f"{label}...", expanded=False) as status:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = round(time.time() - start, 2)
            status.update(label=f"{label} successful ({elapsed}s)", state="complete")
            logging.info("Step succeeded: %s (%.2fs)", label, elapsed)
            return True, result
        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            st.error(str(exc))
            status.update(
                label=f"{label} failed ({elapsed}s)", state="error", expanded=True
            )
            st.session_state.pipeline_failed = True
            logging.exception("Step failed: %s", label)
            return False, None


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

try:
    db_ready = cloud_collection_exists()
except Exception:
    db_ready = False

if db_ready:
    query = st.text_input(
        "Ask a question",
        placeholder="Ask something about the indexed PDFs...",
    )

    if st.button("Get Answer") and query:
        st.session_state.pipeline_failed = False
        logging.info(
            "User Query | %s | %s", time.strftime("%Y-%m-%d %H:%M:%S"), query
        )

        def _retrieve():
            with vector_db_session(embeddings) as store:
                return retrieve_documents(store, query, CONFIG.retrieval_k)

        ok, retrieval_result = run_step("Searching Documents", _retrieve)
        if ok:
            raw_results, query_metrics = retrieval_result
            ok, context_result = run_step("Building Context", build_context, raw_results)
            if ok:
                context, sources = context_result
                prompt = build_prompt(query, context)
                ok, response = run_step("Generating Answer", llm.invoke, prompt)
                if ok:
                    answer = response.content
                    st.markdown("## Answer")
                    st.write(answer)

                    st.markdown("### Retrieval Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Qdrant Retrieval Time",
                            f"{query_metrics.qdrant_retrieval_time:.4f}s",
                        )
                    with col2:
                        st.metric(
                            "Overall Retrieval Time",
                            f"{query_metrics.retrieval_time:.4f}s",
                        )
                    with col3:
                        st.metric("Top-K", CONFIG.retrieval_k)

                    st.markdown("## Source PDFs")
                    for source in sorted(set(sources)):
                        st.write(f"- {source}")

                    with st.expander("Retrieved Chunks"):
                        for index, (doc, score) in enumerate(raw_results, start=1):
                            st.markdown(f"### Chunk {index}")
                            st.write(f"Source: {doc.metadata.get('source_pdf') or doc.metadata.get('filename') or 'Unknown'}")
                            st.write(f"Page: {doc.metadata.get('page')}")
                            st.write(f"Chunk ID: {doc.metadata.get('chunk_id')}")
                            st.write(f"Chunk Index: {doc.metadata.get('chunk_index')}")
                            st.write(f"Score: {score:.4f}")
                            st.write(f"Characters: {doc.metadata.get('char_count')}")
                            st.text(doc.page_content[:500])
                            st.divider()

        if st.session_state.pipeline_failed:
            st.error("Answer generation stopped due to the failed step above.")

else:
    st.info(
        "📭 The Qdrant Cloud collection is empty or unreachable.\n\n"
        "Run the **Colab ingestion notebook** to index your PDFs first, "
        "then reload this page."
    )

if not get_groq_api_key():
    st.warning(
        "`GROQ_API_KEY` is not set. Question answering will fail until it is configured."
    )
