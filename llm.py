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


def get_groq_model() -> str:
    """Retrieve Groq model from Streamlit secrets or environment variables, falling back to CONFIG."""
    try:
        if hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets:
            val = st.secrets["GROQ_MODEL"]
            if val:
                return str(val)
    except Exception:
        pass
    return os.environ.get("GROQ_MODEL") or CONFIG.groq_model_name


@st.cache_resource
def load_llm():
    set_llm_cache(SQLiteCache(database_path=CONFIG.llm_cache_file))
    api_key = get_groq_api_key() or "placeholder_key"
    model_name = get_groq_model()
    return ChatGroq(
        api_key=api_key,
        model=model_name,
    )
