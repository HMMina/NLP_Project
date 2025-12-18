# Báo Cáo Lab 07: Giới Thiệu về Transformers (lab7_intro_transformers.ipynb)

## 📋 Mục Lục
1. [Giải Thích Các Bước Triển Khai](#1-giải-thích-các-bước-triển-khai)
2. [Hướng Dẫn Thực Thi Mã](#2-hướng-dẫn-thực-thi-mã)
3. [Phân Tích Kết Quả](#3-phân-tích-kết-quả)
4. [Thách Thức và Giải Pháp](#4-thách-thức-và-giải-pháp)
5. [Hướng Phát Triển](#5-hướng-phát-triển)

---

## 1. Giải Thích Các Bước Triển Khai

### Task 1: Masked Language Modeling (Fill-Mask)
- **Mục đích**: Hiểu cách hoạt động của mô hình Encoder-only (như BERT) thông qua bài toán điền từ vào chỗ trống.
- **Bước thực hiện**:
  - Sử dụng `pipeline("fill-mask")` từ thư viện `transformers`.
  - Đưa vào câu có chứa token đặc biệt `<mask>` (ví dụ: "Hanoi is the \<mask> of Vietnam.").
  - Mô hình sẽ dự đoán các từ có khả năng điền vào vị trí `<mask>` dựa trên ngữ cảnh hai chiều (trước và sau).
  - **Lý thuyết**: Mô hình Encoder-only (BERT) được huấn luyện để hiểu ngữ cảnh toàn cục, do đó rất giỏi trong việc điền từ bị thiếu.

### Task 2: Causal Language Modeling (Text Generation)
- **Mục đích**: Hiểu cách hoạt động của mô hình Decoder-only (như GPT) thông qua bài toán sinh văn bản.
- **Bước thực hiện**:
  - Sử dụng `pipeline("text-generation")` từ thư viện `transformers`.
  - Cung cấp một đoạn văn bản mồi (prompt), ví dụ: "The best thing about learning NLP is".
  - Mô hình sẽ tự động sinh tiếp các từ tiếp theo dựa trên xác suất.
  - **Lý thuyết**: Mô hình Decoder-only (GPT) được huấn luyện để dự đoán từ tiếp theo (next token prediction) dựa trên các từ đã xuất hiện trước đó (unidirectional), phù hợp cho các tác vụ sáng tạo nội dung.

### Task 3: Sentence Embeddings với BERT
- **Mục đích**: Trích xuất vector biểu diễn ngữ nghĩa của câu (Sentence Embedding) từ mô hình BERT.
- **Bước thực hiện**:
  - Tải mô hình `bert-base-uncased` và tokenizer tương ứng.
  - Tokenize câu đầu vào, thêm padding và truncation để có độ dài cố định.
  - Đưa qua mô hình BERT để lấy `last_hidden_state` (vector biểu diễn của từng token).
  - Thực hiện **Mean Pooling**: Tính trung bình cộng các vector của các token trong câu (lưu ý sử dụng `attention_mask` để loại bỏ các token padding).
  - **Kết quả**: Thu được một vector cố định (kích thước 768) đại diện cho ý nghĩa của toàn bộ câu.

---

## 2. Hướng Dẫn Thực Thi Mã

### 2.1 Yêu Cầu Hệ Thống
```bash
pip install transformers torch
```

### 2.2 Cấu Trúc Thư Mục
```
 notebook/
    Lab07/
       lab7_intro_transformers.ipynb
```

### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook notebook/Lab07/lab7_intro_transformers.ipynb
   ```
2. **Chạy tuần tự các cell**:
   - Cell 1: Task 1 - Fill-Mask với BERT.
   - Cell 2: Trả lời câu hỏi Task 1.
   - Cell 3: Task 2 - Text Generation với GPT-2.
   - Cell 4: Trả lời câu hỏi Task 2.
   - Cell 5: Task 3 - Sentence Embedding với BERT.
   - Cell 6: Trả lời câu hỏi Task 3.

---

## 3. Phân Tích Kết Quả

### 3.1 Task 1: Fill-Mask
- **Input**: "Hanoi is the \<mask> of Vietnam."
- **Dự đoán**:
  - `capital` (score cao nhất): Chính xác về mặt thực tế và ngữ nghĩa.
  - Các từ khác có thể là `heart`, `center`, `city`... tùy thuộc vào ngữ cảnh mà mô hình đã học.
- **Nhận xét**: Mô hình BERT hiểu rất tốt ngữ cảnh hai chiều. Từ "Hanoi" (đứng trước) và "Vietnam" (đứng sau) giúp mô hình xác định chính xác từ cần điền là "capital".

### 3.2 Task 2: Text Generation
- **Input**: "The best thing about learning NLP is"
- **Output**: Một đoạn văn bản tiếp diễn hợp lý, ví dụ: "...that it allows computers to understand human language..."
- **Nhận xét**: Mô hình GPT sinh văn bản trôi chảy, ngữ pháp đúng. Tuy nhiên, nội dung có thể thay đổi mỗi lần chạy do tính chất ngẫu nhiên (sampling) trong quá trình sinh.

### 3.3 Task 3: Sentence Embedding
- **Output**: Một vector có kích thước `(1, 768)`.
- **Ý nghĩa**:
  - Con số 768 tương ứng với `hidden_size` của mô hình `bert-base-uncased`.
  - Vector này chứa thông tin ngữ nghĩa của câu "This is a sample sentence.".
- **Vai trò của Attention Mask**:
  - Nếu không dùng `attention_mask` khi tính trung bình (Mean Pooling), các giá trị 0 của padding token sẽ bị tính vào, làm "loãng" vector biểu diễn của câu, dẫn đến sai lệch ngữ nghĩa.
  - Code đã xử lý đúng bằng cách nhân `last_hidden_state` với `mask_expanded` trước khi tính tổng và chia cho tổng mask.

---

## 4. Thách Thức và Giải Pháp

### 4.1 Thách Thức
- **Kích thước mô hình lớn**: Các mô hình Transformer (BERT, GPT) thường rất nặng, tốn nhiều RAM và thời gian tải.
- **Giới hạn độ dài (Max Sequence Length)**: BERT thường giới hạn 512 tokens. Nếu câu quá dài sẽ bị cắt (truncation), mất thông tin.
- **Padding**: Việc xử lý padding thủ công khi tính pooling khá phức tạp và dễ sai sót.

### 4.2 Giải Pháp
- **Sử dụng Pipeline**: Thư viện `transformers` cung cấp `pipeline` giúp ẩn đi các bước tiền xử lý phức tạp, dễ dàng sử dụng cho người mới.
- **Mean Pooling cẩn thận**: Luôn nhớ sử dụng `attention_mask` để loại bỏ padding khi tính toán thủ công trên output của BERT.
- **DistilBERT**: Sử dụng các phiên bản nhỏ gọn hơn (như DistilBERT) nếu tài nguyên phần cứng hạn chế.

---

## 5. Hướng Phát Triển
- **Fine-tuning**: Thay vì chỉ dùng pre-trained model, có thể fine-tune BERT trên bộ dữ liệu cụ thể (ví dụ: phân loại văn bản y tế, pháp luật) để tăng độ chính xác.
- **Sentence-BERT (SBERT)**: Sử dụng thư viện `sentence-transformers` (được xây dựng dựa trên BERT nhưng tối ưu cho sentence embedding) để tạo vector câu tốt hơn và so sánh độ tương đồng (cosine similarity).
- **Ứng dụng thực tế**:
  - Dùng Sentence Embedding để xây dựng hệ thống tìm kiếm ngữ nghĩa (Semantic Search).
  - Dùng Text Generation để xây dựng Chatbot hoặc công cụ hỗ trợ viết lách.
