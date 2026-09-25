# RAG Evaluation Results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Ragas 0.4.3 / Datasets 4.0.0 |
| Evaluator model                    | Gemini 2.5 Flash |
| Generator model                    | Gemini 2.5 Flash |
| Embedding model                    | BAAI/bge-m3 (dimension 1024) |
| Corpus version/commit              | v1.0.0-final (3 legal policy docs, 5 news articles) |
| Golden dataset size                | 15 Q&A pairs |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.30 (calibrated on in-domain vs out-domain queries) |

## Configurations

- **Config A — dense-only:** Sử dụng BAAI/bge-m3 dense embeddings kết hợp ChromaDB cosine similarity retrieval, lấy top 5 chunks đưa vào generator.
- **Config B — hybrid + RRF:** Kết hợp Dense Semantic Search (ChromaDB) và Lexical Search (BM25 Okapi) trên cùng corpus chunks, gộp thứ hạng bằng Reciprocal Rank Fusion (RRF với k=60), lấy top 5 chunks và kích hoạt PageIndex vectorless fallback khi dense cosine score < 0.30.

Hai config sử dụng cùng golden dataset (15 câu), cùng generator (Gemini 2.5 Flash), evaluator, prompt và `top_k=5`; chỉ thay đổi retrieval strategy.

## Overall scores

| Metric            | Config A (Dense-only) | Config B (Hybrid + RRF) | Delta B−A |
| ----------------- | --------------------: | ----------------------: | --------: |
| Faithfulness      |                 0.862 |                   0.938 |    +0.076 |
| Answer relevance  |                 0.835 |                   0.912 |    +0.077 |
| Context recall    |                 0.794 |                   0.895 |    +0.101 |
| Context precision |                 0.781 |                   0.884 |    +0.103 |
| **Average**       |             **0.818** |               **0.907** | **+0.089**|

## A/B comparison

- **Cấu hình tốt hơn:** Config B (Hybrid + RRF) vượt trội rõ rệt trên tất cả 4 metrics đánh giá, đưa điểm trung bình từ 0.818 lên 0.907 (+8.9%).
- **Evidence:**
  - Đối với các câu hỏi chứa thuật ngữ pháp lý chính xác hoặc điều khoản cụ thể (ví dụ: "Khoản 1 Điều 3", "Điều 9 Luật Du lịch 2017"), BM25 giúp bắt đúng từ khóa gốc, bổ trợ mạnh mẽ cho Dense Search khi câu query quá ngắn hoặc mang tính kỹ thuật.
  - Context Precision tăng mạnh nhất (+0.103) nhờ thuật toán RRF lọc bỏ các chunk gây nhiễu, chỉ đẩy lên đầu các đoạn văn bản mà cả hai bộ tìm kiếm semantic và lexical đều đồng thuận.
  - Cơ chế fallback PageIndex giúp tránh tình trạng rỗng context khi query có khoảng cách ngữ nghĩa lớn với corpus.
- **Trade-off về latency/cost:**
  - Config B có độ trễ tìm kiếm (retrieval latency) trung bình là 48ms so với 28ms của Config A (tăng ~20ms do chạy song song BM25 và tính toán RRF rank fusion). Tuy nhiên thời gian này hoàn toàn chấp nhận được so với tổng thời gian sinh lời gọi LLM (~850ms).
  - Chi phí API của cả hai cấu hình là tương đương nhau do cùng gửi top 5 chunks vào LLM.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Khách du lịch nội địa là gì? | Config A | 0.72 | 0.75 | 0.65 | 0.60 | retrieval | Dense search trả về các đoạn giải thích chung về khách du lịch thay vì điều khoản định nghĩa cụ thể khách nội địa. |
|   2 | Kinh doanh dịch vụ lữ hành là gì? | Config A | 0.78 | 0.80 | 0.70 | 0.68 | retrieval | Chunk size 500 ký tự cắt ngang điều khoản khiến ngữ nghĩa bị phân mảnh trước khi đến LLM. |
|   3 | Định hướng phát triển dịch vụ du lịch tỉnh Bắc Giang? | Config B | 0.85 | 0.84 | 0.78 | 0.75 | generation | Tài liệu quy hoạch dài, nhiều số liệu thống kê khiến câu trả lời của mô hình đôi khi tóm tắt hơi ngắn. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bổ sung Parent Document Retriever hoặc Semantic Chunking theo từng Điều/Khoản luật | Trường hợp câu 1 và câu 2 bị phân mảnh ngữ nghĩa khi chia chunk theo ký tự cố định 500 ký tự | Tăng Context Recall thêm +5% đến +8% cho các câu hỏi tra cứu điều luật | Chạy lại Ragas trên 15 câu golden dataset |
|        2 | Áp dụng Cross-Encoder Reranker (như BGE-Reranker-large) sau bước RRF | RRF phụ thuộc vào ranking số thứ tự, đôi khi chưa đánh giá sâu mức độ tương đồng ngữ nghĩa cấp độ câu | Nâng Context Precision lên trên 0.92 | So sánh metric A/B giữa RRF thuần và RRF + Cross-Encoder |
|        3 | Tinh chỉnh prompt sinh câu trả lời với few-shot examples về trích dẫn điều luật | Trường hợp câu 3 tóm tắt ngắn do prompt chưa chỉ định chi tiết các khía cạnh cần tổng hợp | Tăng Answer Relevance và độ đầy đủ của câu trả lời | Đánh giá qua metric Answer Relevance và kiểm tra thủ công câu trả lời mẫu |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Multi-turn Conversation Memory Query Rewriting | Baseline không dùng memory | Relevance +0.062 trên câu hỏi follow-up | +1 lượt gọi LLM (~150ms) | Cải thiện rõ rệt trải nghiệm người dùng khi hội thoại đa lượt, giúp retrieval nhận diện đầy đủ ý định |
| Citation Highlighting UI with interactive popovers | UI Markdown text thông thường | Tăng tính kiểm chứng (Verifiability) 100% | 0ms latency delta | Người dùng đối soát nguồn ngay lập tức, giảm thiểu rủi ro hallucination trong môi trường pháp lý |
