"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def _normalize_metadata(metadata: dict | None) -> dict:
    normalized = dict(metadata or {})
    normalized.setdefault("url", None)
    return normalized


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0] or []
    documents = response.get("documents", [[]])[0] or []
    metadatas = response.get("metadatas", [[]])[0] or []
    distances = response.get("distances", [[]])[0] or []

    results = []
    for item_id, content, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        results.append(
            {
                "id": item_id,
                "content": content or "",
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": _normalize_metadata(metadata),
                "retrieval_method": "dense",
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    import json

    for result in semantic_search("test query", top_k=3):
        print(json.dumps(result, ensure_ascii=False))
