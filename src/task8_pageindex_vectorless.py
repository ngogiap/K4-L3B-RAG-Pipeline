import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"


def _load_cache() -> dict[str, str]:
    """Load cached document IDs mapping."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read PageIndex cache: %s", e)
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    """Save document IDs mapping to cache."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save PageIndex cache: %s", e)


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key:
        logger.warning("PAGEINDEX_API_KEY is not set. Skipping PageIndex document upload.")
        return

    cache = _load_cache()
    if not STANDARDIZED_DIR.exists():
        logger.warning("Standardized directory %s does not exist yet.", STANDARDIZED_DIR)
        return

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        logger.info("No markdown documents found in %s to upload.", STANDARDIZED_DIR)
        return

    try:
        import pageindex
        client = getattr(pageindex, "Client", None) or getattr(pageindex, "PageIndexClient", None)
        pi_client = client(api_key=api_key) if client else None
    except ImportError:
        logger.warning("pageindex SDK not installed. Skipping upload.")
        return
    except Exception as e:
        logger.warning("Failed to initialize PageIndex client: %s", e)
        return

    updated = False
    for path in md_files:
        rel_key = path.relative_to(STANDARDIZED_DIR).as_posix()
        if rel_key in cache:
            continue

        try:
            content = path.read_text(encoding="utf-8")
            if pi_client and hasattr(pi_client, "upload_document"):
                doc_id = pi_client.upload_document(title=path.stem, content=content)
                if isinstance(doc_id, dict):
                    doc_id = doc_id.get("id") or doc_id.get("doc_id") or str(doc_id)
                cache[rel_key] = str(doc_id)
                updated = True
                logger.info("Uploaded %s to PageIndex with ID %s", rel_key, doc_id)
        except Exception as e:
            logger.error("Failed to upload %s to PageIndex: %s", rel_key, e)

    if updated:
        _save_cache(cache)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key:
        return []

    cache = _load_cache()
    if not cache:
        return []

    try:
        import pageindex
        client_cls = getattr(pageindex, "Client", None) or getattr(pageindex, "PageIndexClient", None)
        if not client_cls:
            return []
        pi_client = client_cls(api_key=api_key)

        doc_ids = list(cache.values())
        raw_results = []
        if hasattr(pi_client, "query"):
            raw_results = pi_client.query(query=query, doc_ids=doc_ids, top_k=top_k)
        elif hasattr(pi_client, "search"):
            raw_results = pi_client.search(query=query, doc_ids=doc_ids, top_k=top_k)

        results: list[dict] = []
        seen_ids = set()
        for rank, item in enumerate(raw_results or [], start=1):
            if isinstance(item, dict):
                item_id = str(item.get("id") or f"pageindex-{rank}")
                content = str(item.get("content") or item.get("text") or "").strip()
                score = float(item.get("score") or (1.0 / (1.0 + rank)))
                meta = item.get("metadata") or {}
            else:
                item_id = f"pageindex-{rank}"
                content = str(item).strip()
                score = 1.0 / (1.0 + rank)
                meta = {}

            if not content or item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            metadata = {
                "source": meta.get("source") or "pageindex_doc",
                "title": meta.get("title") or "PageIndex Result",
                "doc_type": meta.get("doc_type") or "legal",
                "url": meta.get("url"),
                "chunk_index": int(meta.get("chunk_index", 0)),
            }
            results.append({
                "id": item_id,
                "content": content,
                "score": score,
                "metadata": metadata,
                "retrieval_method": "pageindex",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        logger.warning("pageindex_search encountered error: %s", e)
        return []


if __name__ == "__main__":
    upload_documents()

