"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    if top_k <= 0:
        return []
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    best_rank: dict[str, int] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            items.setdefault(item_id, item)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)

    ranked_ids = sorted(
        scores,
        key=lambda item_id: (-scores[item_id], best_rank[item_id], item_id),
    )

    results = []
    for item_id in ranked_ids[:top_k]:
        result = {**items[item_id]}
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
