import streamlit as st
import os
import tempfile
import time
import uuid
import logging
import gc  # Added for strict memory management

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
MAX_FILES         = 50        # Reduced from 1600 to fit within 1GB RAM constraint
MAX_TOTAL_MB      = 200       # Reduced from 10GB to protect container memory
MILVUS_COLLECTION = "rag_docs"

# FIX: Switched from broken text-embedding-004 to active gemini-embedding-001
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"

INDEX_CHUNK_BATCH = 100       # Reduced from 500 to lower memory spikes

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
# CLIENTS
# --------------------------------------------------

@st.cache_resource
def _embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=api_key,
        # FIX: Force the 3072 output to 768 dimensions to optimize Milvus Lite footprint
        output_dimensionality=768
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
            try:
                os.remove(SESSION_DB_PATH)
                logging.info(f"Deleted session DB: {SESSION_DB_PATH}")
                st.success("Vector database deleted.")
            except Exception as e:
                st.error(f"Could not delete database: {e}")
        else:
            st.info("No database found to delete.")
        for k in ("vector_db", "chunk_count", "metrics"):
            st.session_state.pop(k, None)
        st.session_state.pipeline_failed = False
        gc.collect()  # Flush RAM immediately
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

    t0 = time.time()
    ok, vector_db = run_step("Opening / Creating Vector Database", _open_milvus, SESSION_DB_PATH)
    metrics["milvus_init_time"] = round(time.time() - t0, 2)

    if ok:
        total_chunks  = 0
        
        for uf in uploaded_files:
            if st.session_state.pipeline_failed:
                break

            def _load(uf=uf):
                tmp_dir = tempfile.gettempdir()
                tmp = os.path.join(tmp_dir, f"{uuid.uuid4()}_{uf.name}")
                with open(tmp, "wb") as fh:
                    fh.write(uf.getbuffer())
                
                try:
                    docs = PyPDFLoader(tmp).load()
                    for doc in docs:
                        doc.metadata["source_pdf"] = uf.name
                        doc.metadata["session_id"] = st.session_state.session_id
                    return docs
                finally:
                    # FIX: Safely remove the temporary file to keep disk space low
                    if os.path.exists(tmp):
                        os.remove(tmp)

            t0 = time.time()
            ok_load, docs = run_step(f"Loading: {uf.name}", _load)
            
            if ok_load and docs:
                # Chunk and immediately commit to Milvus
                chunks = splitter.split_documents(docs)
                total_chunks += len(chunks)
                
                # Batch upload chunks to limit memory spike
                for i in range(0, len(chunks), INDEX_CHUNK_BATCH):
                    batch = chunks[i:i + INDEX_CHUNK_BATCH]
                    vector_db.add_documents(batch)
                
                # FIX: Force Python to completely drop the data references from RAM
                del docs
                del chunks
                gc.collect()

        st.session_state.chunk_count = total_chunks
        st.success(f"Successfully processed {pdf_count} files ({total_chunks} chunks)!")
