"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


import re


CORPUS: list[dict] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _normalize_metadata(metadata: dict | None) -> dict:
    normalized = dict(metadata or {})
    normalized.setdefault("url", None)
    return normalized


def _load_corpus_from_vectorstore() -> list[dict]:
    """Best-effort load of indexed chunks, used when CORPUS is not prefilled."""
    try:
        from .task4_chunking_indexing import get_collection

        response = get_collection().get(include=["documents", "metadatas"])
    except Exception:
        return []

    ids = response.get("ids") or []
    documents = response.get("documents") or []
    metadatas = response.get("metadatas") or []
    corpus = []
    for item_id, content, metadata in zip(ids, documents, metadatas):
        if item_id and content and metadata:
            corpus.append(
                {
                    "id": item_id,
                    "content": content,
                    "metadata": _normalize_metadata(metadata),
                }
            )
    return corpus


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(item["content"]) for item in corpus]
    return BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    corpus = CORPUS or _load_corpus_from_vectorstore()
    if not corpus:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(query_tokens)
    if not any(float(score) > 0 for score in scores):
        query_set = set(query_tokens)
        scores = [
            len(query_set.intersection(_tokenize(item["content"]))) / len(query_set)
            for item in corpus
        ]
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (float(scores[index]), corpus[index]["id"]),
        reverse=True,
    )

    results = []
    seen_ids: set[str] = set()
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        item = corpus[index]
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": _normalize_metadata(item["metadata"]),
                "retrieval_method": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    import json

    for result in lexical_search("test query", top_k=3):
        print(json.dumps(result, ensure_ascii=False))
