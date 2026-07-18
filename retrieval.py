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
            f"""
        SOURCE: {source}
        PAGE: {doc.metadata.get("page")}
        CHUNK: {doc.metadata.get("chunk_id")}
        {doc.page_content}
        """
        )
    return "\n\n".join(context_parts), sources


def build_prompt(query: str, context: str) -> str:
    return f"""
You are an expert document question-answering assistant.

Answer the question using ONLY the retrieved context provided below.

Rules:
- Do NOT use prior knowledge.
- Do NOT infer missing information.
- Do NOT guess.
- Do NOT fabricate any information.
- Do NOT complete or generate partial values.
- Never invent hashes, hexadecimal values, CVEs, registry keys, IP addresses, file paths, URLs, filenames, or technical identifiers.
If the answer can be directly determined from the retrieved context,
answer it.

Only reply

Information not found in retrieved documents.

when the retrieved context truly does not contain the answer.

- If multiple retrieved chunks contain conflicting information, only report what is explicitly stated in the retrieved context and do not attempt to resolve the conflict.
If the retrieved context contains information from multiple documents,
combine the information from all relevant documents before answering.
Retrieved Context:
{context}

Question:
{query}

Answer:
"""
