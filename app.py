import streamlit as st
import os
import tempfile
import time
import uuid
import logging
import gc  # Used for strict memory management

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # Switched from Milvus to Chroma
from langchain_groq import ChatGroq

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Multi PDF RAG System", layout="wide")
st.title("Multi-PDF Semantic RAG System (Chroma Edition)")

VECTOR_DB_ROOT    = "./chroma_db"
LOG_FILE          = "./rag.log"
MAX_FILES         = 50        
MAX_TOTAL_MB      = 200       

# Active, supported embedding endpoint
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
INDEX_CHUNK_BATCH = 100       

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

if "pipeline_failed" not in st.session_state:
    st.session_state.pipeline_failed = False

if "chunk_tracking" not in st.session_state:
    st.session_state.chunk_tracking = {}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

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
# CLIENTS (Cached Resource Components)
# --------------------------------------------------

@st.cache_resource
def _embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=api_key,
        output_dimensionality=768 # Match standard vector dimensional scale
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
# CHROMA RECTIFICATION COMPONENT
# --------------------------------------------------

@st.cache_resource
def _init_chroma():
    """Initializes a persistent Chroma vector instance isolated by session."""
    persist_path = os.path.join(VECTOR_DB_ROOT, st.session_state.session_id)
    return Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory=persist_path
    )

vector_db = _init_chroma()

# --------------------------------------------------
# SIDEBAR INFO & STATISTICS
# --------------------------------------------------

with st.sidebar:
    st.markdown("### 📊 Session Diagnostics")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}…`")

    if "chunk_count" in st.session_state:
        st.metric("Total Indexed Chunks", f"{st.session_state.chunk_count:,}")
    
    if st.session_state.chunk_tracking:
        st.markdown("**PDF Index Profiles:**")
        for filename, info in st.session_state.chunk_tracking.items():
            st.caption(f"📄 *{filename}*")
            st.write(f"└ Chunks: `{info['count']}` | Time: `{info['time']}s`")

    if st.button("Clear Data & Reset Instance"):
        persist_path = os.path.join(VECTOR_DB_ROOT, st.session_state.session_id)
        if os.path.exists(persist_path):
            import shutil
            shutil.rmtree(persist_path)
            st.success("Chroma vector indexes wiped.")
        
        for k in ["chunk_count", "chunk_tracking", "pipeline_failed"]:
            if k in st.session_state:
                del st.session_state[k]
        
        st.cache_resource.clear()
        gc.collect()
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

# --------------------------------------------------
# CONSTRUCT PIPELINE EXECUTION
# --------------------------------------------------

if uploaded_files and upload_valid and st.button("Process PDFs"):
    st.session_state.pipeline_failed = False
    pdf_count = len(uploaded_files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    total_chunks = 0
    tracking_metrics = {}

    for uf in uploaded_files:
        if st.session_state.pipeline_failed:
            break

        # 1. Load File to Disk Temporary Target
        def _load(file_obj=uf):
            tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{file_obj.name}")
            with open(tmp, "wb") as fh:
                fh.write(file_obj.getbuffer())
            try:
                docs = PyPDFLoader(tmp).load()
                for doc in docs:
                    doc.metadata["source_pdf"] = file_obj.name
                    doc.metadata["session_id"] = st.session_state.session_id
                return docs
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        ok_load, docs = run_step(f"Parsing PDF Structure: {uf.name}", _load)
        
        if ok_load and docs:
            # 2. Extract Text Segments (Chunk Size metrics tracking)
            chunks = splitter.split_documents(docs)
            n_chunks = len(chunks)
            del docs
            gc.collect()

            if n_chunks == 0:
                continue

            # 3. Chroma Ingestion Execution Tracking
            def _index(chunks_to_embed=chunks):
                t_start = time.time()
                for i in range(0, len(chunks_to_embed), INDEX_CHUNK_BATCH):
                    batch = chunks_to_embed[i:i + INDEX_CHUNK_BATCH]
                    vector_db.add_documents(batch)
                return round(time.time() - t_start, 2)

            ok_idx, execution_time = run_step(f"Chroma Processing Matrix: {uf.name}", _index)
            
            if ok_idx:
                total_chunks += n_chunks
                tracking_metrics[uf.name] = {
                    "count": n_chunks,
                    "time": execution_time
                }
            
            del chunks
            gc.collect()

    if not st.session_state.pipeline_failed:
        st.session_state.chunk_count = total_chunks
        st.session_state.chunk_tracking = tracking_metrics
        st.success(f"Ingested {pdf_count} PDFs into Chroma Vector DB!")

# --------------------------------------------------
# RETRIEVAL INTERFACE (With Latency Verification)
# --------------------------------------------------

st.divider()
st.markdown("### 🔍 Real-Time Semantic Engine Query")
user_query = st.text_input("Ask a question about your documents:")

if user_query:
    if "chunk_count" not in st.session_state or st.session_state.chunk_count == 0:
        st.warning("Please upload and process your PDFs before executing queries.")
    else:
        # Trace exact Chroma indexing query resolution latency
        t_retrieval_start = time.time()
        retrieved_docs = vector_db.similarity_search(user_query, k=4)
        retrieval_latency = round(time.time() - t_retrieval_start, 4)
        
        # Display explicit telemetry tracking metric metrics
        st.info(f"⏱️ **Chroma Retrieval Latency**: `{retrieval_latency} seconds` | Matches Extracted: `4 chunks`")
        
        # Build LLM RAG Context Payload
        context_payload = "\n\n".join([f"[Source: {d.metadata.get('source_pdf', 'Unknown')}]: {d.page_content}" for d in retrieved_docs])
        
        prompt_structure = f"""You are a helpful assistant. Use the following context pieces to answer the query. If you do not know the answer, say you don't know.

Context:
{context_payload}

User Query: {user_query}
Answer:"""

        with st.spinner("Generating LLM Response..."):
            try:
                response = llm.invoke(prompt_structure)
                st.markdown("####System Response:")
                st.write(response.content)
                
                # Context expansion view layout
                with st.expander("Inspect Retrieved Source Chunks"):
                    for idx, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {idx+1}** - *Source File: {doc.metadata.get('source_pdf')}*")
                        st.caption(doc.page_content)
                        st.divider()
            except Exception as e:
                st.error(f"LLM Engine inference execution failed: {e}")
