import os

import streamlit as st
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_groq import ChatGroq

from config import CONFIG


@st.cache_resource
def load_llm():
    set_llm_cache(SQLiteCache(database_path=CONFIG.llm_cache_file))
    api_key = os.environ.get("GROQ_API_KEY") or "placeholder_key"
    return ChatGroq(
        api_key=api_key,
        model=CONFIG.groq_model_name,
    )
