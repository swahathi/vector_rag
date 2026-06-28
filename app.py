import streamlit as st
import os
import tempfile
import time
import uuid
import logging
#test
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus
from langchain_groq import ChatGroq

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Multi PDF RAG System", layout="wide")
st.title("Multi-PDF Semantic RAG System")

VECTOR_DB_ROOT    = "./vector_db"
LOG_FILE          = "./rag.log"
MAX_FILES         = 1600
MAX_TOTAL_MB      = 10_000
MILVUS_COLLECTION = "rag_docs"

GEMINI_EMBED_MODEL = "models/text-embedding-004"

# Gemini's embed_documents batches at 100 texts per API call automatically.
# We control how many *chunks* we hand to add_documents() per iteration to cap
# peak RAM (chunks queued in Python memory), not to control the API batch size.
INDEX_CHUNK_BATCH = 500

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --------------------------------------------------
# SESSION ISOLATION
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Milvus Lite persists to a single .db file per session.
os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
SESSION_DB_PATH = os.path.join(VECTOR_DB_ROOT, f"{st.session_state.session_id}.db")

# --------------------------------------------------
# PIPELINE STATE
# --------------------------------------------------

if "pipeline_failed" not in st.session_state:
    st.session_state.pipeline_failed = False


def run_step(label: str, func, *args, **kwargs):
    """Execute func inside a Streamlit status block. Returns (ok, result)."""
    if st.session_state.pipeline_failed:
        return False, None
    with st.status(f"{label}...", expanded=False) as status:
        t0 = time.time()
        try:
            result  = func(*args, **kwargs)
            elapsed = round(time.time() - t0, 2)
            status.update(label=f"{label} ✓ ({elapsed}s)", state="complete", expanded=False)
            logging.info(f"Step OK: {label} ({elapsed}s)")
            return True, result
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            st.error(str(exc))
            status.update(label=f"{label} ✗ ({elapsed}s)", state="error", expanded=True)
            st.session_state.pipeline_failed = True
            logging.error(f"Step FAIL: {label} — {exc}")
            return False, None


# --------------------------------------------------
# CLIENTS  (created once per process; no vector caching, no LLM caching)
# --------------------------------------------------

@st.cache_resource
def _embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Single GoogleGenerativeAIEmbeddings instance.
    - embed_documents() uses RETRIEVAL_DOCUMENT task type by default.
    - embed_query()     uses RETRIEVAL_QUERY    task type by default.
    No task_type kwarg needed here; the LangChain wrapper applies the correct
    default automatically for each call path.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=api_key,
    )


@st.cache_resource
def _llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)


embeddings = _embeddings()
llm        = _llm()


# --------------------------------------------------
# MILVUS HELPERS
# --------------------------------------------------

def _open_milvus(db_path: str) -> Milvus:
    """
    Open (or create) a Milvus Lite vector store at *db_path*.
    drop_old=False  → existing data is preserved across Streamlit reruns.
    auto_id=True    → Milvus generates primary keys; no client-side ID tracking.
    """
    return Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": db_path},
        auto_id=True,
        drop_old=False,
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Session")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}…`")

    if "chunk_count" in st.session_state:
        st.caption(f"Indexed chunks: {st.session_state.chunk_count:,}")

    if st.button("Clear My Data"):
        if os.path.exists(SESSION_DB_PATH):
            os.remove(SESSION_DB_PATH)
            logging.info(f"Deleted session DB: {SESSION_DB_PATH}")
            st.success("Vector database deleted.")
        else:
            st.info("No database found to delete.")
        for k in ("vector_db", "chunk_count", "metrics"):
            st.session_state.pop(k, None)
        st.session_state.pipeline_failed = False
        st.rerun()

# --------------------------------------------------
# FILE UPLOAD + VALIDATION
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files", type=["pdf"], accept_multiple_files=True
)
st.caption(f"Limit: {MAX_FILES} files · {MAX_TOTAL_MB:,} MB total per session.")

upload_valid = True
if uploaded_files:
    total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
    if len(uploaded_files) > MAX_FILES:
        st.error(f"Too many files ({len(uploaded_files)}). Limit is {MAX_FILES}.")
        upload_valid = False
    if total_size_mb > MAX_TOTAL_MB:
        st.error(f"Upload too large ({total_size_mb:.1f} MB). Limit is {MAX_TOTAL_MB} MB.")
        upload_valid = False
    if upload_valid:
        st.caption(f"{len(uploaded_files)} file(s) · {total_size_mb:.1f} MB — within limits.")

# --------------------------------------------------
# BUILD VECTOR DB
# --------------------------------------------------

if uploaded_files and upload_valid and st.button("Process PDFs"):

    pipeline_start = time.time()
    st.session_state.pipeline_failed = False

    pdf_count       = len(uploaded_files)
    dataset_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
    db_existed      = os.path.exists(SESSION_DB_PATH)
    metrics: dict   = {}
    splitter        = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # ── Open / create Milvus ──────────────────────────────────────────────────
    t0 = time.time()
    ok, vector_db = run_step("Opening / Creating Vector Database", _open_milvus, SESSION_DB_PATH)
    metrics["milvus_init_time"] = round(time.time() - t0, 2)

    if ok:
        total_chunks  = 0
        load_elapsed  = 0.0
        chunk_elapsed = 0.0
        index_elapsed = 0.0

        # ── Per-PDF loop: load → chunk → index → release ──────────────────────
        # Memory design: we never hold all pages + all chunks simultaneously.
        # Each PDF is fully processed and its intermediate data freed before
        # the next PDF begins.
        for uf in uploaded_files:
            if st.session_state.pipeline_failed:
                break

            # Load ─────────────────────────────────────────────────────────────
            def _load(uf=uf):
                tmp = os.path.join(tempfile.gettempdir(), uf.name)
                with open(tmp, "wb") as fh:
                    fh.write(uf.getbuffer())
                docs = PyPDFLoader(tmp).load()
                for doc in docs:
                    doc.metadata["source_pdf"] = uf.name
                    doc.metadata["session_id"] = st.session_state.session_id
                return docs

            t0 = time.time()
            ok, docs = run_step(f"Loading: {uf.name}", _load)
            load_elapsed += time.time() - t0
            if not ok:
                break

            # Chunk ────────────────────────────────────────────────────────────
            def _chunk(docs=docs):
                return splitter.split_documents(docs)

            t0 = time.time()
            ok, chunks = run_step(f"Chunking: {uf.name}", _chunk)
            chunk_elapsed += time.time() - t0
            del docs          # pages no longer needed
            if not ok:
                break

            # Index ────────────────────────────────────────────────────────────
            # We split chunks into INDEX_CHUNK_BATCH slices (default 500).
            # This bounds Python-side RAM while Gemini's own embed_documents()
            # sub-batches each slice at 100 texts per API call automatically.
            def _index(chunks=chunks, vdb=vector_db):
                n = 0
                for i in range(0, len(chunks), INDEX_CHUNK_BATCH):
                    vdb.add_documents(chunks[i : i + INDEX_CHUNK_BATCH])
                    n += min(INDEX_CHUNK_BATCH, len(chunks) - i)
                return n

            t0 = time.time()
            ok, n = run_step(f"Indexing:  {uf.name}", _index)
            index_elapsed += time.time() - t0
            del chunks        # chunks freed before next PDF
            if not ok:
                break

            total_chunks += n
            logging.info(f"'{uf.name}': {n} chunks indexed.")

        metrics["load_time"]  = round(load_elapsed,  2)
        metrics["chunk_time"] = round(chunk_elapsed, 2)
        metrics["index_time"] = round(index_elapsed, 2)

        if not st.session_state.pipeline_failed:
            total_time = round(time.time() - pipeline_start, 2)
            db_size_mb = os.path.getsize(SESSION_DB_PATH) / (1024 * 1024) if os.path.exists(SESSION_DB_PATH) else 0.0

            st.session_state.vector_db   = vector_db
            st.session_state.chunk_count = st.session_state.get("chunk_count", 0) + total_chunks
            st.session_state.metrics     = metrics

            logging.info(f"Pipeline done: {total_chunks} chunks / {pdf_count} PDFs / {total_time}s")
            st.success(f"Indexed {total_chunks:,} chunks from {pdf_count} file(s).")

            # ── Metrics dashboard ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Experiment Metrics")

            init_t = metrics["milvus_init_time"]
            if db_existed:
                st.info(f"**Database status:** Loaded existing database — init: {init_t:.2f}s")
            else:
                st.success(f"**Database status:** Created new database — init: {init_t:.2f}s")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Milvus initialization",    f"{metrics['milvus_init_time']:.2f}s")
                st.metric("PDF loading",              f"{metrics['load_time']:.2f}s")
                st.metric("Chunking",                 f"{metrics['chunk_time']:.2f}s")
            with c2:
                st.metric("Indexing (embed + insert)", f"{metrics['index_time']:.2f}s")
                st.metric("Total processing time",     f"{total_time:.2f}s")
                st.metric("Vector database size",      f"{db_size_mb:.2f} MB")

            st.markdown("**Run summary**")
            st.write(f"PDFs processed: **{pdf_count}**")
            st.write(f"Dataset size: **{dataset_size_mb:.2f} MB**")
            st.write(f"Total chunks indexed: **{total_chunks:,}**")

# --------------------------------------------------
# RECONNECT TO EXISTING SESSION DB ON RERUN
# --------------------------------------------------

if "vector_db" not in st.session_state and os.path.exists(SESSION_DB_PATH):
    try:
        st.session_state.vector_db = _open_milvus(SESSION_DB_PATH)
        logging.info(f"Reconnected to existing session DB: {SESSION_DB_PATH}")
    except Exception as exc:
        logging.error(f"Failed to reconnect to session DB: {exc}")
        st.warning(f"Could not load your existing database: {exc}")

# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

if "vector_db" in st.session_state:
    vector_db = st.session_state.vector_db
    query     = st.text_input("Ask a question", placeholder="Ask something about the uploaded PDFs…")

    if st.button("Get Answer") and query:
        st.session_state.pipeline_failed = False
        logging.info(f"Query received: {query}")

        # Retrieve ─────────────────────────────────────────────────────────────
        # similarity_search_with_relevance_scores encodes `query` via
        # embed_query() which automatically uses RETRIEVAL_QUERY task type.
        def _retrieve():
            return vector_db.similarity_search_with_relevance_scores(query, k=8)

        ok, raw_results = run_step("Searching Documents", _retrieve)

        # Build context ────────────────────────────────────────────────────────
        if ok:
            def _build_context():
                ctx_parts, sources = [], set()
                for doc, _ in raw_results:
                    src = doc.metadata.get("source_pdf", "Unknown")
                    sources.add(src)
                    ctx_parts.append(f"SOURCE: {src}\n\n{doc.page_content}")
                return "\n\n---\n\n".join(ctx_parts), sources

            ok, (context, sources) = run_step("Building Context", _build_context)

        # Generate answer ──────────────────────────────────────────────────────
        if ok:
            prompt = (
                "You are a document question-answering assistant.\n"
                "Answer ONLY using the provided context.\n"
                "If the answer is not present, say: "
                '"Information not found in retrieved documents."\n\n'
                f"Context:\n{context}\n\nQuestion:\n{query}"
            )

            def _answer():
                return llm.invoke(prompt).content

            ok, answer = run_step("Generating Answer", _answer)

            if ok:
                logging.info("Answer generated successfully.")
                st.markdown("## Answer")
                st.write(answer)

                st.markdown("## Source PDFs")
                for src in sorted(sources):
                    st.write(f"- {src}")

                with st.expander("Retrieved Chunks"):
                    for idx, (doc, score) in enumerate(raw_results, 1):
                        st.markdown(f"### Chunk {idx}")
                        st.write(f"**Source:** {doc.metadata.get('source_pdf')}")
                        st.write(f"**Score:** {score:.4f}")
                        st.text(doc.page_content[:500])
                        st.divider()

        if st.session_state.pipeline_failed:
            st.error("Pipeline stopped due to the failed step above.")

else:
    st.info("Upload PDFs and click **Process PDFs** to begin.")