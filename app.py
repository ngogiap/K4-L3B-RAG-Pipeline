import os
import re
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation, call_llm
from src.task9_retrieval_pipeline import retrieve, SCORE_THRESHOLD, DEFAULT_TOP_K


load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Hệ Thống Trợ Lý RAG Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design and citation highlighting
st.markdown(
    """
    <style>
    .source-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: #adb5bd;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .badge-hybrid {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-pageindex {
        background-color: #f3e5f5;
        color: #4a148c;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-dense {
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-bm25 {
        background-color: #fff3e0;
        color: #e65100;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .citation-tag {
        color: #0284c7;
        font-weight: 700;
        background-color: #e0f2fe;
        padding: 1px 5px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def contextualize_query(history: list[dict], current_query: str) -> str:
    """Tạo câu hỏi độc lập từ lịch sử hội thoại (Conversation Memory - Bonus)."""
    if not history:
        return current_query

    # Chỉ lấy tối đa 3 lượt hội thoại gần nhất
    recent_history = history[-6:]
    history_str = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = (
        "Dựa trên lịch sử trò chuyện dưới đây và câu hỏi mới của người dùng (có thể có đại từ thay thế như 'nó', 'điều đó', 'thế còn...'), "
        "hãy viết lại câu hỏi thành một câu hỏi độc lập, đầy đủ ngữ cảnh để tìm kiếm tài liệu. "
        "Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm.\n\n"
        f"Lịch sử:\n{history_str}\n"
        f"Câu hỏi mới: {current_query}\n"
        "Câu hỏi độc lập:"
    )
    try:
        rewritten = call_llm(
            "Bạn là chuyên gia chuyển hóa câu hỏi theo ngữ cảnh cuộc hội thoại.",
            prompt,
        ).strip()
        return rewritten if rewritten else current_query
    except Exception:
        return current_query


def format_citation_markdown(text: str) -> str:
    """Highlight các tag [Document X] hoặc [X] trong câu trả lời."""
    # Thay thế [Document X] thành HTML badge đẹp mắt
    highlighted = re.sub(
        r"\[Document\s*(\d+)\]",
        r"<span class='citation-tag'>📄 [Tài liệu \1]</span>",
        text,
        flags=re.IGNORECASE,
    )
    return highlighted


def render_source_badge(method: str) -> str:
    """Render badge HTML theo retrieval method."""
    method_lower = (method or "").lower()
    if "pageindex" in method_lower:
        return "<span class='badge-pageindex'>🌲 PageIndex Fallback</span>"
    elif "hybrid" in method_lower:
        return "<span class='badge-hybrid'>⚡ Hybrid (RRF)</span>"
    elif "dense" in method_lower:
        return "<span class='badge-dense'>🔍 Dense Semantic</span>"
    elif "bm25" in method_lower:
        return "<span class='badge-bm25'>📄 Lexical BM25</span>"
    return f"<span class='badge-hybrid'>{method}</span>"


# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Cấu hình Pipeline")
    st.caption("Hệ thống RAG Pipeline nhóm K4-L3B")

    llm_provider = os.getenv("LLM_PROVIDER", "gemini").upper()
    st.info(f"🤖 **Mô hình LLM**: `{llm_provider}`")

    st.subheader("Tham số Tìm kiếm")
    top_k = st.slider("Số lượng Chunks (top_k)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    score_threshold = st.slider(
        "Ngưỡng Dense Fallback (threshold)",
        min_value=0.0,
        max_value=1.0,
        value=float(SCORE_THRESHOLD),
        step=0.05,
        help="Nếu cosine similarity của Dense retrieval < ngưỡng này, hệ thống sẽ kích hoạt PageIndex vectorless fallback.",
    )
    use_reranking = st.toggle("Kích hoạt RRF Reranking (Hybrid)", value=True)
    use_memory = st.toggle(
        "Kích hoạt Conversation Memory (Bonus +2đ)",
        value=True,
        help="Tự động liên kết ngữ cảnh các câu hỏi tiếp theo trong phiên chat.",
    )

    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Lab 8: Hybrid Retrieval, Citation & Evaluation")


# Main Page Header
st.title("📚 Trợ Lý Tra Cứu Thông Tin RAG Pipeline")
st.markdown(
    "Hệ thống hỏi đáp tài liệu chính sách và tin tức tích hợp **Dense Semantic Search + BM25 Lexical + Reciprocal Rank Fusion (RRF) + PageIndex Fallback**."
)

# Suggested Prompts when conversation is empty
if not st.session_state.messages:
    st.markdown("##### 💡 Câu hỏi gợi ý:")
    cols = st.columns(3)
    suggestions = [
        "Phạm vi điều chỉnh của Luật Du lịch 2017?",
        "Tài nguyên du lịch gồm những loại nào?",
        "Phương án phát triển dịch vụ tỉnh Bắc Giang?",
    ]
    for col, prompt_text in zip(cols, suggestions):
        if col.button(prompt_text, use_container_width=True):
            st.session_state["pending_query"] = prompt_text
            st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            # Hiển thị câu hỏi đã được làm giàu ngữ cảnh (nếu có)
            if message.get("contextualized_query"):
                st.caption(f"🔍 *Đã tối ưu câu hỏi tìm kiếm: \"{message['contextualized_query']}\"*")

            # Hiển thị câu trả lời với citation styling
            formatted_answer = format_citation_markdown(message["content"])
            st.markdown(formatted_answer, unsafe_allow_html=True)

            # Hiển thị Sources & Citation Details
            sources = message.get("sources", [])
            retrieval_source = message.get("retrieval_source", "none")

            if sources:
                with st.expander(f"📑 Nguồn tài liệu tham khảo ({len(sources)} trích đoạn - {retrieval_source.upper()})", expanded=False):
                    for idx, chunk in enumerate(sources, start=1):
                        meta = chunk.get("metadata", {})
                        title = meta.get("title", f"Tài liệu {idx}")
                        source_file = meta.get("source", "Không rõ nguồn")
                        method = chunk.get("retrieval_method", "hybrid")
                        score = chunk.get("score", 0.0)
                        url = meta.get("url")

                        url_link = f" | [🔗 Xem bài gốc]({url})" if url else ""
                        badge_html = render_source_badge(method)

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div><b>[Document {idx}] {title}</b> {badge_html}</div>
                                <div style="font-size: 0.85rem; color: #6c757d; margin-top: 4px;">
                                    📁 File: <code>{source_file}</code> | Độ khớp (Score): <b>{score:.4f}</b>{url_link}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.popover(f"👁️ Xem chi tiết nội dung [Document {idx}]"):
                            st.write(chunk.get("content", ""))
            elif retrieval_source == "none":
                st.warning("⚠️ Không tìm thấy bằng chứng liên quan trong tập dữ liệu.")


# User input
query = st.chat_input("Nhập câu hỏi của bạn về chính sách hoặc tin tức...")
if "pending_query" in st.session_state and st.session_state["pending_query"]:
    query = st.session_state.pop("pending_query")


if query:
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Xử lý Conversation Memory (Bonus)
    contextualized_q = query
    if use_memory and len(st.session_state.messages) > 1:
        with st.spinner("Đang phân tích ngữ cảnh hội thoại..."):
            contextualized_q = contextualize_query(st.session_state.messages[:-1], query)

    # 3. Thực hiện Retrieval & Generation
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                # Gọi trực tiếp generate_with_citation
                result = generate_with_citation(contextualized_q, top_k=top_k)
                answer = result.get("answer", "Không thể tạo câu trả lời.")
                sources = result.get("sources", [])
                retrieval_source = result.get("retrieval_source", "none")
            except Exception as e:
                answer = f"Đã xảy ra lỗi trong quá trình xử lý: {e}\n\n*(Lưu ý: Nếu đồng đội chưa hoàn thành Task 1-7, ChromaDB hoặc BM25 index có thể chưa sẵn sàng)*"
                sources = []
                retrieval_source = "none"

        # Hiển thị query mở rộng nếu khác query ban đầu
        if contextualized_q != query:
            st.caption(f"🔍 *Đã tối ưu câu hỏi tìm kiếm: \"{contextualized_q}\"*")

        # Hiển thị câu trả lời
        formatted_answer = format_citation_markdown(answer)
        st.markdown(formatted_answer, unsafe_allow_html=True)

        # Hiển thị Sources
        if sources:
            with st.expander(f"📑 Nguồn tài liệu tham khảo ({len(sources)} trích đoạn - {retrieval_source.upper()})", expanded=True):
                for idx, chunk in enumerate(sources, start=1):
                    meta = chunk.get("metadata", {})
                    title = meta.get("title", f"Tài liệu {idx}")
                    source_file = meta.get("source", "Không rõ nguồn")
                    method = chunk.get("retrieval_method", "hybrid")
                    score = chunk.get("score", 0.0)
                    url = meta.get("url")

                    url_link = f" | [🔗 Xem bài gốc]({url})" if url else ""
                    badge_html = render_source_badge(method)

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div><b>[Document {idx}] {title}</b> {badge_html}</div>
                            <div style="font-size: 0.85rem; color: #6c757d; margin-top: 4px;">
                                📁 File: <code>{source_file}</code> | Độ khớp (Score): <b>{score:.4f}</b>{url_link}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.popover(f"👁️ Xem chi tiết nội dung [Document {idx}]"):
                        st.write(chunk.get("content", ""))
        elif retrieval_source == "none":
            st.warning("⚠️ Không tìm thấy bằng chứng liên quan trong tập dữ liệu.")

    # 4. Lưu vào session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "contextualized_query": contextualized_q if contextualized_q != query else None,
    })

