import logging
import os

from dotenv import load_dotenv

from .contracts import GenerationResult
from .task9_retrieval_pipeline import retrieve


load_dotenv()

logger = logging.getLogger(__name__)

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Bạn là trợ lý ảo giải đáp thông tin từ tài liệu chính sách và bài viết chính thức.
Quy tắc trả lời:
1. CHỈ sử dụng thông tin có trong các đoạn context được cung cấp. Tuyệt đối không tự suy diễn hoặc bịa đặt.
2. Mỗi thông tin, khẳng định đưa ra PHẢI có trích dẫn nguồn tương ứng bằng ký hiệu [Document X] (ví dụ: [Document 1], [Document 2]).
3. Nếu các đoạn context được cung cấp KHÔNG chứa đủ bằng chứng hoặc không liên quan đến câu hỏi, bạn PHẢI từ chối trả lời một cách an toàn:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có."
4. Trình bày bằng tiếng Việt rõ ràng, ngắn gọn, súc tích và có cấu trúc."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (giảm hiện tượng Lost-in-the-Middle).

    Hàm không làm biến đổi (non-mutating) danh sách gốc.
    """
    if not chunks:
        return []
    if len(chunks) <= 2:
        return [dict(c) for c in chunks]

    front = chunks[::2]
    back = chunks[1::2]
    reordered = front + back[::-1]
    return [dict(c) for c in reordered]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label để LLM tạo citation kiểm chứng được."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown")
        content = chunk.get("content", "").strip()
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi Gemini, OpenAI hoặc Anthropic theo cấu hình LLM_PROVIDER trong .env."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL).strip()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        model_name = model or "gemini-3.8-flash"
        try:
            from google import genai
            from google.genai import types


            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                ),
            )
            return response.text or ""
        except ImportError:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=api_key)
            g_model = legacy_genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
            )
            response = g_model.generate_content(
                user_message,
                generation_config={"temperature": TEMPERATURE, "top_p": TOP_P},
            )
            return response.text or ""

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        model_name = model or "gpt-4o-mini"
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        model_name = model or "claude-3-5-sonnet-20241022"
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            max_tokens=1500,
        )
        return response.content[0].text or ""

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Thực hiện retrieval, reordering, format context và sinh câu trả lời có citation.

    Output tuân thủ contract GenerationResult:
        - answer: str
        - sources: list[SearchResult]
        - retrieval_source: 'hybrid' | 'pageindex' | 'none'
    """
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception as e:
        logger.warning("Retrieve failed with error: %s", e)
        chunks = []

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Xác định retrieval_source theo contract ("hybrid" | "pageindex" | "none")
    first_method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source = "pageindex" if first_method == "pageindex" else "hybrid"

    try:
        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        user_message = f"Dưới đây là các tài liệu liên quan:\n\n{context}\n\n---\nCâu hỏi của người dùng: {query}\n\nHãy trả lời câu hỏi dựa trên các tài liệu trên kèm theo trích dẫn [Document X]."
        answer = call_llm(SYSTEM_PROMPT, user_message)

        if not answer or not answer.strip():
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

        return {
            "answer": answer.strip(),
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }
    except Exception as e:
        logger.error("Generation error: %s", e)
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có do sự cố kết nối tới mô hình AI.",
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }


if __name__ == "__main__":
    print(generate_with_citation("test query"))

