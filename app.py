import gc
import logging
import os
import time
import uuid

import streamlit as st

from chunking import build_text_splitter
from config import CONFIG
from embeddings import load_embeddings
from evaluation import ExperimentSnapshot
from ingestion import ingest_uploaded_files
from llm import load_llm
from logging_utils import append_jsonl_record, configure_logging
from retrieval import build_context, build_prompt, retrieve_documents
from vectordb import (
    clear_session_db,
    get_db_size_mb,
    load_manifest,
    open_vector_db,
    vector_db_exists,
)


st.set_page_config(page_title=CONFIG.page_title, layout=CONFIG.page_layout)
st.title(CONFIG.app_title)

configure_logging(CONFIG.log_file)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "pipeline_failed" not in st.session_state:
    st.session_state.pipeline_failed = False


def run_step(label, func, *args, **kwargs):
    with st.status(f"{label}...", expanded=False) as status:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = round(time.time() - start, 2)
            status.update(
                label=f"{label} successful ({elapsed}s)",
                state="complete",
                expanded=False,
            )
            logging.info("Step succeeded: %s (%.2fs)", label, elapsed)
            return True, result
        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            st.error(str(exc))
            status.update(
                label=f"{label} failed ({elapsed}s)",
                state="error",
                expanded=True,
            )
            st.session_state.pipeline_failed = True
            logging.exception("Step failed: %s", label)
            return False, None


embeddings = load_embeddings()
llm = load_llm()

with st.sidebar:
    st.markdown("### Session")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

    if "chunk_count" in st.session_state:
        st.caption(f"Indexed chunks: {st.session_state.chunk_count}")

    manifest = load_manifest(st.session_state.session_id)
    indexed_files = manifest.get("indexed_files", {})
    st.caption(f"Indexed PDFs in session: {len(indexed_files)}")

    if st.button("Clear My Data"):
        # Drop every reference to the Chroma client (and anything else tied
        # to this session's DB) *before* deleting its files. On Windows,
        # Chroma's persistent index (data_level0.bin) is memory-mapped, and
        # the OS refuses to unlink a file that's still open/mapped anywhere
        # in the process. Deleting first and popping session_state after
        # (the previous order) meant the client was still alive when
        # shutil.rmtree ran, causing PermissionError: [WinError 32].
        for key in ("vector_db", "chunk_count", "metrics", "last_ingestion_summary"):
            st.session_state.pop(key, None)
        gc.collect()

        clear_session_db(st.session_state.session_id)
        logging.info("Cleared session DB for %s", st.session_state.session_id)
        st.session_state.pipeline_failed = False
        st.success("Vector database deleted.")
        st.rerun()


uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True,
)
st.caption(f"Limit: {CONFIG.max_files} files, {CONFIG.max_total_mb} MB total per session.")

upload_valid = True
if uploaded_files:
    total_size_mb = sum(file.size for file in uploaded_files) / (1024 * 1024)
    if len(uploaded_files) > CONFIG.max_files:
        st.error(
            f"Too many files: {len(uploaded_files)} uploaded, "
            f"limit is {CONFIG.max_files}. Please remove some files."
        )
        upload_valid = False
    if total_size_mb > CONFIG.max_total_mb:
        st.error(
            f"Upload too large: {total_size_mb:.1f} MB, "
            f"limit is {CONFIG.max_total_mb} MB. Please upload fewer/smaller files."
        )
        upload_valid = False
    if upload_valid:
        st.caption(
            f"{len(uploaded_files)} file(s), {total_size_mb:.1f} MB total - within limits."
        )


def render_indexing_results(metrics, summary, db_already_existed):
    st.success(
        f"Successfully indexed {summary.total_chunks} chunks from "
        f"{metrics.processed_files} new file(s)."
    )
    if metrics.skipped_duplicates:
        st.info(f"Skipped {metrics.skipped_duplicates} duplicate file(s).")
    if metrics.failed_files:
        st.warning(f"{metrics.failed_files} file(s) failed during indexing.")

    st.markdown("---")
    st.subheader("Experiment Metrics")

    if db_already_existed:
        st.info(
            f"Database status: Loaded existing database - Initialization time: "
            f"{metrics.chroma_init_time:.2f}s"
        )
    else:
        st.success(
            f"Database status: Created new database - Initialization time: "
            f"{metrics.chroma_init_time:.2f}s"
        )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chroma initialization", f"{metrics.chroma_init_time:.2f}s")
        st.metric("PDF loading", f"{metrics.load_time:.2f}s")
        st.metric("Chunking", f"{metrics.chunk_time:.2f}s")
        st.metric("Embedding", f"{metrics.embedding_time:.2f}s")
    with col2:
        st.metric("Vector insertion", f"{metrics.vector_insertion_time:.2f}s")
        st.metric("Total processing time", f"{metrics.total_indexing_time:.2f}s")
        st.metric("Vector database size", f"{metrics.db_size_mb:.2f} MB")
        st.metric("Total chunks indexed", str(metrics.total_chunks))

    st.markdown("**Run summary**")
    st.write(f"PDFs uploaded: **{metrics.pdf_count}**")
    st.write(f"Dataset size: **{metrics.dataset_size_mb:.2f} MB**")
    st.write(f"Chunk size: **{metrics.chunk_size} characters**")
    st.write(f"Chunk overlap: **{metrics.chunk_overlap} characters**")

    with st.expander("File-level processing results"):
        for item in summary.file_results:
            line = f"{item.file_name}: {item.status}"
            if item.chunk_count:
                line += f" ({item.chunk_count} chunks)"
            if item.error:
                line += f" - {item.error}"
            st.write(line)


if uploaded_files and upload_valid and st.button("Process PDFs"):
    st.session_state.pipeline_failed = False
    st.session_state.last_ingestion_summary = None
    splitter = build_text_splitter()
    db_already_existed = vector_db_exists(st.session_state.session_id)

    def _open_vector_db():
        return open_vector_db(st.session_state.session_id, embeddings)

    init_start = time.time()
    ok, vector_db = run_step("Opening / Creating Vector Database", _open_vector_db)

    if ok:
        def _ingest():
            return ingest_uploaded_files(
                uploaded_files=uploaded_files,
                session_id=st.session_state.session_id,
                vector_db=vector_db,
                splitter=splitter,
            )

        ok, ingestion_result = run_step("Processing PDFs in batches", _ingest)
        if ok:
            metrics, summary = ingestion_result
            metrics.chroma_init_time = time.time() - init_start
            metrics.db_size_mb = get_db_size_mb(st.session_state.session_id)
            st.session_state.vector_db = vector_db
            st.session_state.chunk_count = st.session_state.get("chunk_count", 0) + summary.total_chunks
            st.session_state.metrics = metrics.as_dict()
            st.session_state.last_ingestion_summary = summary
            append_jsonl_record(
                CONFIG.experiment_log_file,
                {
                    "timestamp": time.time(),
                    "session_id": st.session_state.session_id,
                    "metrics": metrics.as_dict(),
                },
            )
            render_indexing_results(metrics, summary, db_already_existed)


if "vector_db" not in st.session_state and vector_db_exists(st.session_state.session_id):
    try:
        st.session_state.vector_db = open_vector_db(
            st.session_state.session_id,
            embeddings,
        )

    except Exception as exc:
        logging.exception("Database corrupted. Creating new database.")

        clear_session_db(st.session_state.session_id)

        st.session_state.vector_db = open_vector_db(
            st.session_state.session_id,
            embeddings,
        )

        st.info("Previous database was corrupted and has been recreated.")


if "vector_db" in st.session_state:
    vector_db = st.session_state.vector_db
    query = st.text_input(
        "Ask a question",
        placeholder="Ask something about the uploaded PDFs...",
    )

    if st.button("Get Answer") and query:
        st.session_state.pipeline_failed = False
        logging.info("Query: %s", query)

        ok, retrieval_result = run_step(
            "Searching Documents",
            retrieve_documents,
            vector_db,
            query,
            CONFIG.retrieval_k,
        )
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
                        st.metric("Chroma Retrieval Time", f"{query_metrics.chroma_retrieval_time:.4f}s")
                    with col2:
                        st.metric("Overall Retrieval Time", f"{query_metrics.retrieval_time:.4f}s")
                    with col3:
                        st.metric("Configured Chunk Size", f"{CONFIG.chunk_size} chars")

                    st.markdown("## Source PDFs")
                    for source in sources:
                        st.write(f"- {source}")

                    with st.expander("Retrieved Chunks"):
                        for index, (doc, score) in enumerate(raw_results, start=1):
                            st.markdown(f"### Chunk {index}")
                            st.write(f"Source: {doc.metadata.get('source_pdf')}")
                            st.write(f"Score: {score:.4f}")
                            st.write(f"Chunk Size (Actual): {len(doc.page_content)} characters")
                            st.text(doc.page_content[:500])
                            st.divider()
        if st.session_state.pipeline_failed:
            st.error("Answer generation stopped due to the failed step above.")
else:
    st.info("Upload PDFs and click Process PDFs.")


if not os.environ.get("GROQ_API_KEY"):
    st.warning("`GROQ_API_KEY` is not set in the environment. Question answering will fail until it is configured.")

_ = ExperimentSnapshot(session_id=st.session_state.session_id)