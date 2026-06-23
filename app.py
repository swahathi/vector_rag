
import streamlit as st
import os
import shutil
import tempfile
import time
import uuid
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore

from langchain_community.vectorstores import Chroma

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from langchain_groq import ChatGroq

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Multi PDF RAG System",
    layout="wide"
)

st.title("Multi-PDF Semantic RAG System")

EMBED_CACHE_DIR = "./embedding_cache"
VECTOR_DB_ROOT  = "./vector_db"
LOG_FILE        = "./rag.log"

MAX_FILES      = 1600
MAX_TOTAL_MB   = 10000

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# SESSION ISOLATION
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_DB_DIR = os.path.join(VECTOR_DB_ROOT, st.session_state.session_id)

# --------------------------------------------------
# STEP RUNNER
# --------------------------------------------------

if "pipeline_failed" not in st.session_state:
    st.session_state.pipeline_failed = False


def run_step(label, func, *args, **kwargs):
    if st.session_state.pipeline_failed:
        return False, None

    with st.status(f"{label}...", expanded=False) as status:
        start = time.time()
        try:
            result  = func(*args, **kwargs)
            elapsed = round(time.time() - start, 2)
            status.update(
                label=f"{label} successful ({elapsed}s)",
                state="complete",
                expanded=False,
            )
            logging.info(f"Step succeeded: {label} ({elapsed}s)")
            return True, result

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            st.error(str(e))
            status.update(
                label=f"{label} failed ({elapsed}s)",
                state="error",
                expanded=True,
            )
            st.session_state.pipeline_failed = True
            logging.error(f"Step failed: {label} - {e}")
            return False, None

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def load_embeddings():
    base_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = LocalFileStore(EMBED_CACHE_DIR)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace="all-MiniLM-L6-v2"
    )
    return cached_embeddings

# --------------------------------------------------
# LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():
    set_llm_cache(SQLiteCache(database_path="langchain_cache.db"))
    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.1-8b-instant"
    )
    return llm


embeddings = load_embeddings()
llm        = load_llm()

# --------------------------------------------------
# SIDEBAR — session info + single clear button
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Session")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

    if "chunk_count" in st.session_state:
        st.caption(f"Indexed chunks: {st.session_state.chunk_count}")

    if st.button("Clear My Data"):
        if os.path.exists(SESSION_DB_DIR):
            shutil.rmtree(SESSION_DB_DIR)
            logging.info(f"Cleared session DB: {SESSION_DB_DIR}")
            st.success("Vector database deleted.")
        else:
            st.info("No database found to delete.")

        for key in ("vector_db", "chunk_count", "metrics"):
            st.session_state.pop(key, None)

        st.session_state.pipeline_failed = False
        st.rerun()

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

st.caption(f"Limit: {MAX_FILES} files, {MAX_TOTAL_MB} MB total per session.")

# --------------------------------------------------
# VALIDATE UPLOAD SIZE / COUNT
# --------------------------------------------------

upload_valid = True

if uploaded_files:
    total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)

    if len(uploaded_files) > MAX_FILES:
        st.error(
            f"Too many files: {len(uploaded_files)} uploaded, "
            f"limit is {MAX_FILES}. Please remove some files."
        )
        upload_valid = False

    if total_size_mb > MAX_TOTAL_MB:
        st.error(
            f"Upload too large: {total_size_mb:.1f} MB, "
            f"limit is {MAX_TOTAL_MB} MB. Please upload fewer/smaller files."
        )
        upload_valid = False

    if upload_valid:
        st.caption(
            f"{len(uploaded_files)} file(s), {total_size_mb:.1f} MB total — within limits."
        )

# --------------------------------------------------
# BUILD VECTOR DB
# --------------------------------------------------

if uploaded_files and upload_valid:

    if st.button("Process PDFs"):

        experiment_start = time.time()
        st.session_state.pipeline_failed = False

        # Collect per-step wall-clock timings here
        metrics = {}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        pdf_count      = len(uploaded_files)
        dataset_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)

        # Detect DB status BEFORE opening so we can report it accurately
        db_already_existed = os.path.exists(SESSION_DB_DIR)

        # ---- Open / create Chroma DB ----
        def _open_vector_db():
            return Chroma(
                persist_directory=SESSION_DB_DIR,
                embedding_function=embeddings
            )

        t0 = time.time()
        ok, vector_db = run_step("Opening/ Creating Vector Database...", _open_vector_db)
        metrics["chroma_init_time"] = round(time.time() - t0, 2)

        if ok:

            # ---- Phase 1: Load all PDFs ----
            def _load_all_pdfs():
                all_docs = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(
                        tempfile.gettempdir(),
                        uploaded_file.name
                    )
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    loader = PyPDFLoader(temp_path)
                    docs   = loader.load()

                    for doc in docs:
                        doc.metadata["source_pdf"] = uploaded_file.name
                        doc.metadata["session_id"] = st.session_state.session_id

                    all_docs.append(docs)
                return all_docs

            t0 = time.time()
            ok, all_docs = run_step(f"Loading {pdf_count} PDF(s)", _load_all_pdfs)
            metrics["load_time"] = round(time.time() - t0, 2)

            if ok:
                logging.info(
                    f"Loading phase complete: {pdf_count} file(s), "
                    f"{sum(len(d) for d in all_docs)} pages"
                )

            # ---- Phase 2: Chunk all PDFs ----
            if ok:
                def _chunk_all(all_docs=all_docs):
                    all_chunks = []
                    for docs in all_docs:
                        all_chunks.extend(splitter.split_documents(docs))
                    return all_chunks

                t0 = time.time()
                ok, all_chunks = run_step(f"Chunking {pdf_count} PDF(s)", _chunk_all)
                metrics["chunk_time"] = round(time.time() - t0, 2)

                if ok:
                    logging.info(
                        f"Chunking phase complete: {len(all_chunks)} chunks "
                        f"from {pdf_count} file(s)"
                    )

            # ---- Phase 3: Index all chunks ----
            if ok:
                def _index_all(all_chunks=all_chunks):
                    BATCH_SIZE = 1000
                    for i in range(0, len(all_chunks), BATCH_SIZE):
                        batch = all_chunks[i:i + BATCH_SIZE]
                        vector_db.add_documents(batch)
                    return len(all_chunks)

                t0 = time.time()
                ok, total_chunks = run_step(f"Indexing {pdf_count} PDF(s)", _index_all)
                metrics["index_time"] = round(time.time() - t0, 2)

                if ok:
                    logging.info(
                        f"Indexing phase complete: {total_chunks} chunks indexed"
                    )

            if ok and not st.session_state.pipeline_failed:

                total_time = round(time.time() - experiment_start, 2)

                db_size_mb = 0
                if os.path.exists(SESSION_DB_DIR):
                    db_size_mb = sum(
                        os.path.getsize(os.path.join(root, file))
                        for root, dirs, files in os.walk(SESSION_DB_DIR)
                        for file in files
                    ) / (1024 * 1024)

                st.session_state.vector_db   = vector_db
                st.session_state.chunk_count = (
                    st.session_state.get("chunk_count", 0) + total_chunks
                )
                st.session_state.metrics = metrics

                st.success(
                    f"Successfully indexed {total_chunks} chunks from {pdf_count} file(s)."
                )

                st.markdown("---")
                st.subheader("Experiment Metrics")

                # --- Database status ---
                init_time = metrics.get('chroma_init_time', 0)
                if db_already_existed:
                    st.info(
                        f"**Database status:** Loaded existing database — Initialization time: {init_time:.2f}s",
                        icon="📂"
                    )
                else:
                    st.success(
                        f"**Database status:** Created new database — Initialization time: {init_time:.2f}s"
                    )

                # --- Per-step timing grid ---
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Chroma initialization",
                        f"{metrics.get('chroma_init_time', 0):.2f}s"
                    )
                    st.metric(
                        "PDF loading",
                        f"{metrics.get('load_time', 0):.2f}s"
                    )
                    st.metric(
                        "Chunking",
                        f"{metrics.get('chunk_time', 0):.2f}s"
                    )

                with col2:
                    st.metric(
                        "Indexing (embed + insert)",
                        f"{metrics.get('index_time', 0):.2f}s"
                    )
                    st.metric(
                        "Total processing time",
                        f"{total_time:.2f}s"
                    )
                    st.metric(
                        "Vector database size",
                        f"{db_size_mb:.2f} MB"
                    )

                # --- Run summary ---
                st.markdown("**Run summary**")
                st.write(f"PDFs processed: **{pdf_count}**")
                st.write(f"Dataset size: **{dataset_size_mb:.2f} MB**")
                st.write(f"Total chunks indexed: **{total_chunks}**")

# --------------------------------------------------
# LOAD EXISTING SESSION VECTOR DB ON RERUN
# --------------------------------------------------

if (
    "vector_db" not in st.session_state
    and os.path.exists(SESSION_DB_DIR)
):
    try:
        st.session_state.vector_db = Chroma(
            persist_directory=SESSION_DB_DIR,
            embedding_function=embeddings
        )
        logging.info(f"Loaded existing session vector DB: {SESSION_DB_DIR}")
    except Exception as e:
        logging.error(f"Failed to load session vector DB: {e}")
        st.warning(f"Could not load your existing database: {e}")

# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

if "vector_db" in st.session_state:

    vector_db = st.session_state.vector_db

    query = st.text_input(
        "Ask a question",
        placeholder="Ask something about the uploaded PDFs..."
    )

    if st.button("Get Answer") and query:

        st.session_state.pipeline_failed = False
        logging.info(f"Query: {query}")

        # ---- Step 1: Retrieve chunks ----
        def _retrieve():
            return vector_db.similarity_search_with_relevance_scores(
                query,
                k=8
            )

        ok, raw_results = run_step("Searching Documents", _retrieve)

        # ---- Step 2: Build context ----
        def _build_context():
            context = ""
            sources = set()

            for doc, score in raw_results:
                source = doc.metadata.get("source_pdf", "Unknown")
                sources.add(source)
                context += f"""

SOURCE: {source}

{doc.page_content}

"""
            return context, sources

        ok, ctx_result = run_step("Building Context", _build_context)

        if ok:
            context, sources = ctx_result

            prompt = f"""
You are a document question-answering assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:

"Information not found in retrieved documents."

Context:
{context}

Question:
{query}
"""

            # ---- Step 3: Generate answer ----
            def _generate_answer():
                response = llm.invoke(prompt)
                return response.content

            ok, answer = run_step("Generating Answer", _generate_answer)

            if ok:
                logging.info("Answer generated successfully")

                st.markdown("## Answer")
                st.write(answer)

                st.markdown("## Source PDFs")
                for source in sources:
                    st.write(f"- {source}")

                with st.expander("Retrieved Chunks"):
                    for idx, (doc, score) in enumerate(raw_results):
                        st.markdown(f"### Chunk {idx+1}")
                        st.write(f"Source: {doc.metadata.get('source_pdf')}")
                        st.write(f"Score: {score:.4f}")
                        st.text(doc.page_content[:500])
                        st.divider()

        if st.session_state.pipeline_failed:
            st.error("Answer generation stopped due to the failed step above.")

else:

    st.info(
        "Upload PDFs and click Process PDFs."
    )
