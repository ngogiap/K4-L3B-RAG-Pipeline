import os
import logging
from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

logger = logging.getLogger(__name__)

ENV_THRESHOLD = os.getenv("SCORE_THRESHOLD", "").strip()
SCORE_THRESHOLD = float(ENV_THRESHOLD) if ENV_THRESHOLD else 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    # 1. Chạy semantic_search và lexical_search để lấy candidate chunks
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)

    # 2. Lấy best cosine score gốc từ dense results (không dùng RRF score)
    best_dense_score = float(dense[0]["score"]) if dense else 0.0

    # 3. Nếu score dưới threshold, thử PageIndex fallback
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as e:
            logger.warning("PageIndex fallback failed: %s. Continuing with hybrid results.", e)

    # 4. Khi dense đạt threshold hoặc fallback thất bại/rỗng:
    # Fuse hai danh sách bằng RRF đúng một lần nếu use_reranking=True
    if use_reranking:
        if not dense and not sparse:
            return []
        fused = rerank_rrf([dense, sparse], top_k=top_k)
        return fused[:top_k]

    return dense[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)

