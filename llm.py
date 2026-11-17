import os

import streamlit as st
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_groq import ChatGroq

from config import CONFIG


def get_groq_api_key() -> str:
    """Retrieve Groq API key from Streamlit secrets or environment variables."""
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            val = st.secrets["GROQ_API_KEY"]
            if val:
                return str(val)
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY") or ""


@st.cache_resource
def load_llm():
    set_llm_cache(SQLiteCache(database_path=CONFIG.llm_cache_file))
    api_key = get_groq_api_key() or "placeholder_key"
    return ChatGroq(
        api_key=api_key,
        model=CONFIG.groq_model_name,
    )
