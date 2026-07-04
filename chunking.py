from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CONFIG


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )