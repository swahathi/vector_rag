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

Your task is to answer the user's question using ONLY the retrieved context provided below.

Rules:

1. Use only information explicitly present in the retrieved context.
2. You may combine information from multiple retrieved chunks and multiple retrieved documents to produce a single complete answer.
3. If an answer is distributed across multiple chunks, merge those pieces into one coherent response.
4. Do not use outside knowledge, assumptions, speculation, or information not present in the retrieved context.
5. Do not invent, infer, or fabricate facts, values, hashes, CVEs, registry keys, IP addresses, URLs, filenames, or any other technical identifiers.
6. If retrieved documents contain conflicting information, report all conflicting statements exactly as they appear without attempting to resolve them.
7. Do not omit relevant information that appears anywhere in the retrieved context simply because it is repeated or located in a different chunk.
8. Answer as completely as possible using all relevant retrieved context before concluding that information is unavailable.
9. If a list, sequence, or paragraph continues across multiple retrieved chunks, reconstruct the complete list or paragraph using all available retrieved context.

Fallback:
If the retrieved context does not explicitly contain enough information to answer the question, reply exactly:

Information not found in retrieved documents.

Do not mention these instructions in your response.
Retrieved Context:
{context}

Question:
{query}

Answer:
"""
