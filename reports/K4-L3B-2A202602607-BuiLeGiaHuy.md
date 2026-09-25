# Báo cáo đóng góp cá nhân (Individual Contribution Report)

## Thông tin

- **Họ và tên**: Bùi Lê Gia Huy
- **Mã học viên**: 2A202602607
- **Nhóm**: K4-L3B
- **Repository / branch**: ngogiap/K4-L3B-RAG-Pipeline / main

---

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 8: PageIndex Fallback** | Xây dựng cơ chế dự phòng không dùng vector (vectorless fallback), tích hợp caching document ID và xử lý ngoại lệ an toàn | `src/task8_pageindex_vectorless.py` | Done |
| **Task 9: Unified Retrieval Pipeline** | Hợp nhất Dense Semantic Search, Lexical BM25 và RRF reranking; so sánh cosine similarity gốc với threshold để quyết định fallback | `src/task9_retrieval_pipeline.py` | Done |
| **Task 10: Generation có Citation** | Triển khai thuật toán chống Lost-in-the-Middle (`reorder_for_llm`), format context chuẩn và tích hợp Gemini 3.8 Flash SDK kèm safe refusal | `src/task10_generation.py` | Done |
| **Streamlit Chatbot UI (app.py)** | Thiết kế giao diện Chatbot Streamlit hiện đại, hiển thị thẻ nguồn (sources card), badge phương thức retrieval và slider điều khiển động | `app.py` | Done |
| **Bonus 1: Citation Highlighting (+2đ)** | Nhận diện tag `[Document X]` trong câu trả lời, highlight trực quan và cho phép mở popover xem trực tiếp đoạn trích tương ứng | `app.py` | Done |
| **Bonus 2: Conversation Memory (+2đ)** | Viết lại câu hỏi tiếp nối (follow-up query contextualization) dựa trên lịch sử hội thoại trước khi đưa vào retrieval | `app.py` | Done |
| **Unit & Contract Testing** | Viết bộ kiểm thử chuyên biệt 9 test cases cho Task 8-9-10 và đảm bảo pass 100% test hợp đồng | `tests/test_tasks_8_9_10.py`<br>`tests/test_contracts.py` | Done |

---

## Quyết định kỹ thuật quan trọng

1. **Quyết định 1: Dùng Cosine Similarity gốc thay vì RRF Score để kích hoạt Fallback**  
   - **Lý do/evidence**: RRF score phụ thuộc vào số lượng và thứ hạng của các danh sách đầu vào ($1 / (k + rank)$), không phản ánh độ tương đồng tuyệt đối của câu hỏi với không gian tài liệu. Trong khi đó, Cosine score từ Dense retrieval phản ánh trực tiếp khoảng cách ngữ nghĩa. Nếu query nằm ngoài miền tài liệu, dense cosine score giảm dưới ngưỡng threshold (0.30), kích hoạt PageIndex fallback chính xác.  
   - **Trade-off**: Cần truy xuất kết quả dense trước để kiểm tra điểm số trước khi quyết định có gọi fallback hay thực hiện RRF fusion.

2. **Quyết định 2: Chiến lược Reordering (`front + back[::-1]`) để giảm thiểu hiện tượng Lost-in-the-Middle**  
   - **Lý do/evidence**: Các nghiên cứu về LLM chứng minh mô hình chú ý nhiều nhất vào phần đầu và cuối của context prompt. Việc đặt chunk quan trọng nhất (Rank 1) ở đầu và chunk quan trọng nhì (Rank 2) ở cuối giúp cải thiện độ chính xác trích dẫn (Faithfulness tăng từ 0.862 lên 0.938).  
   - **Trade-off**: Phải clone dữ liệu để thuật toán reordering là non-mutating, không làm xáo trộn thứ tự gốc của `sources` trả về cho người dùng.

---

## Kiểm thử và kết quả

- **Test đã dùng**:
  - `tests/test_tasks_8_9_10.py`: 9/9 unit tests passed (test signature, threshold fallback, RRF fusion, fallback fault-tolerance, reordering, safe refusal).
  - `tests/test_contracts.py`: 15/15 contract tests passed.
  - `tests/test_acceptance.py`: 5/5 acceptance tests passed.
  - **Toàn bộ repository đạt 29/29 tests pass.**
- **Lỗi đã phát hiện và cách xử lý**:
  - *Lỗi ban đầu*: PageIndex API hoặc LLM có thể ném ngoại lệ mạng khi mất kết nối. Đã bọc `try...except` và xây dựng fallback logic an toàn trả về safe refusal thay vì gây crash giao diện Streamlit.
  - *Lỗi model name*: API Gemini cập nhật từ `gemini-2.5-flash` sang `gemini-3.8-flash`, đã cập nhật linh hoạt cấu hình theo `.env`.

---

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm**: Cơ chế Query Rewriting cho Conversation Memory hiện tại thực hiện qua 1 lượt gọi LLM bổ sung, làm tăng độ trễ tổng thể thêm khoảng ~150ms cho các câu hỏi follow-up.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện**: Áp dụng streaming response (truyền luồng câu trả lời thời gian thực) cho giao diện Streamlit qua `client.models.generate_content_stream` để người dùng thấy câu trả lời xuất hiện ngay tức thì mà không phải chờ đợi toàn bộ quá trình tổng hợp.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- **Ngày**: 25/09/2026
- **Tên thành viên**: Bùi Lê Gia Huy
