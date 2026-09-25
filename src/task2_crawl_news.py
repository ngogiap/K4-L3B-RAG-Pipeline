"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
# Crawl4AI writes its browser profile and SQLite cache beneath this directory.
# Keep it in the repository rather than the user's home directory so the task
# works in restricted environments as well.
CRAWL4AI_BASE_DIR = Path(__file__).parent.parent / "data" / "_crawl4ai"
os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(CRAWL4AI_BASE_DIR))

ARTICLE_URLS = [
    "https://baobacgiang.vn/bg2/moi-nhat/quang-ba-cac-diem-du-lich-van-hoa-postid413098.bbg",
    "https://baobacgiang.vn/tieng-goi-non-cao-tay-yen-tu.bbg",
    "https://www.baobacgiang.vn/bg/dulichbg/diem-den/403571/khu-du-lich-sinh-thai-suoi-mo-diem-den-hap-dan-cua-du-khach.html",
    "https://baobacgiang.vn/bg2/dulichbg/ngoai-vai-thieu-bac-giang-con-nhung-loat-dac-san-hap-dan-nao.bbg",
    "https://www.baobacgiang.vn/bg2/luc-ngan/khai-mac-chuong-trinh-du-lich-vai-thieu-luc-ngan-tinh-hoa-trai-cay-viet--postid419650.bbg",
]


async def crawl_article(url: str) -> dict:
    """Crawl one public article and return the landing-zone schema.

    Crawl4AI has returned either a string or a MarkdownGenerationResult for
    ``result.markdown`` across versions, so both forms are handled here.
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    return _article_from_result(url, result)


def _article_from_result(url: str, result: Any) -> dict:
    """Normalize one Crawl4AI result to the required landing-zone schema."""

    if not result.success:
        raise RuntimeError(getattr(result, "error_message", None) or "Crawler returned no result")

    metadata: dict[str, Any] = result.metadata or {}
    markdown = result.markdown or ""
    if not isinstance(markdown, str):
        markdown = (
            getattr(markdown, "fit_markdown", None)
            or getattr(markdown, "raw_markdown", None)
            or str(markdown)
        )
    markdown = markdown.strip()
    if not markdown:
        raise RuntimeError("Crawler returned empty article content")

    title = str(metadata.get("title") or "Untitled article").strip()
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    from crawl4ai import AsyncWebCrawler

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncWebCrawler() as crawler:
        for index, url in enumerate(ARTICLE_URLS, 1):
            try:
                result = await crawler.arun(url=url)
                article = _article_from_result(url, result)
                output = DATA_DIR / f"article_{index:02d}.json"
                output.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"Saved: {output}")
            except Exception as error:
                print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
