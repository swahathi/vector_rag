from __future__ import annotations

import time

from metrics import QueryMetrics


def retrieve_documents(vector_db, query: str, k: int):
    metrics = QueryMetrics()
    start = time.time()
    results = vector_db.similarity_search_with_relevance_scores(query, k=20)
    metrics.chroma_retrieval_time = time.time() - start
    metrics.retrieval_time = metrics.chroma_retrieval_time
    return results, metrics


def build_context(raw_results):
    context_parts = []
    sources = set()
    for doc, _score in raw_results:
        source = doc.metadata.get("source_pdf", "Unknown")
        sources.add(source)
        context_parts.append(
            f"SOURCE: {source}\n\n{doc.page_content}"
        )
    return "\n\n".join(context_parts), sources


def build_prompt(query: str, context: str) -> str:
    return f"""
You are a document question-answering assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:

"Information not found in retrieved documents."

Context:
{context}

Question:
{query}
"""
