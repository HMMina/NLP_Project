# Báo Cáo Lab 08: Dependency Parsing với spaCy (lab8.ipynb)

## 📋 Mục Lục
1. [Giải Thích Các Bước Triển Khai](#1-giải-thích-các-bước-triển-khai)
2. [Hướng Dẫn Thực Thi Mã](#2-hướng-dẫn-thực-thi-mã)
3. [Phân Tích Kết Quả](#3-phân-tích-kết-quả)
4. [Thách Thức và Giải Pháp](#4-thách-thức-và-giải-pháp)
5. [Hướng Phát Triển](#5-hướng-phát-triển)

---

## 1. Giải Thích Các Bước Triển Khai

### Task 1: Làm quen với spaCy và Dependency Parsing
- **Mục đích**: Hiểu cách spaCy phân tích cấu trúc ngữ pháp của câu thông qua Dependency Parsing.
- **Bước thực hiện**:
  - Tải mô hình ngôn ngữ tiếng Anh `en_core_web_md`.
  - Phân tích câu ví dụ: "The quick brown fox jumps over the lazy dog."
  - Sử dụng `displacy` để trực quan hóa cây phụ thuộc.
  - **Lý thuyết**: Dependency Parsing biểu diễn cấu trúc câu dưới dạng các mối quan hệ phụ thuộc (dependency relations) giữa các từ, trong đó một từ là **Head** (cha) và từ kia là **Dependent** (con).

### Task 2: Phân tích Quan hệ Phụ thuộc
- **Mục đích**: Trích xuất thông tin chi tiết về nhãn quan hệ (dep), từ cha (head), và từ loại (POS).
- **Bước thực hiện**:
  - Duyệt qua từng token trong câu.
  - In ra các thuộc tính: `text`, `dep_` (nhãn quan hệ), `head.text` (từ cha), `head.pos_` (từ loại của cha), `children` (các từ con).
  - **Ví dụ 1**: Trong cụm "brown fox", "fox" là Head, "brown" là Dependent với quan hệ `amod` (adjectival modifier).
  - **Ví dụ 2**: Câu "Apple is looking at buying U.K. startup for $1 billion" cho thấy khả năng xử lý các thực thể tên riêng (Apple, U.K.) và số tiền ($1 billion) của spaCy.

### Task 3: Trích xuất Thông tin (Information Extraction)
- **Mục đích**: Ứng dụng Dependency Parsing để rút trích các thông tin có cấu trúc.
- **Bài toán 1: Trích xuất bộ ba Chủ ngữ - Động từ - Tân ngữ (SVO)**
  - Tìm các động từ (`VERB`).
  - Với mỗi động từ, tìm con có nhãn `nsubj` (chủ ngữ) và `dobj` (tân ngữ trực tiếp).
- **Bài toán 2: Trích xuất Danh từ và Tính từ bổ nghĩa**
  - Tìm các danh từ (`NOUN`).
  - Tìm con có nhãn `amod` (tính từ bổ nghĩa).

### Task 4: Các Bài Tập Nâng Cao
- **Bài 1: Tìm Động từ chính (ROOT)**
  - Tìm token có nhãn `dep_ == "ROOT"`. Đây thường là động từ chính của câu.
- **Bài 2: Trích xuất Cụm Danh từ (Noun Chunks)**
  - **Thủ công**: Tìm danh từ và các từ bổ nghĩa (`det`, `amod`, `compound`), sau đó ghép lại.
  - **Tự động**: So sánh với thuộc tính `doc.noun_chunks` có sẵn của spaCy.
- **Bài 3: Tìm đường đi đến gốc (Path to Root)**
  - Từ một token bất kỳ, duyệt ngược lên `head` cho đến khi gặp `ROOT`. Giúp hiểu cấp độ phụ thuộc của từ trong cây cú pháp.

---

## 2. Hướng Dẫn Thực Thi Mã

### 2.1 Yêu Cầu Hệ Thống
```bash
pip install spacy
python -m spacy download en_core_web_md
```

### 2.2 Cấu Trúc Thư Mục
```
 notebook/
    Lab08/
       lab8.ipynb
```

### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook notebook/Lab08/lab8.ipynb
   ```
2. **Chạy tuần tự các cell**:
   - Cell 1-2: Import spaCy và trực quan hóa cây phụ thuộc.
   - Cell 3: Trả lời câu hỏi về quan hệ phụ thuộc.
   - Cell 4-5: Phân tích chi tiết các token.
   - Cell 6: Trích xuất SVO (Subject-Verb-Object).
   - Cell 7: Trích xuất Danh từ - Tính từ.
   - Cell 8-10: Các bài tập nâng cao (Tìm ROOT, Noun Chunks, Path to Root).

---

## 3. Phân Tích Kết Quả

### 3.1 Phân tích Cây Phụ Thuộc
- **Câu**: "The quick brown fox jumps over the lazy dog."
- **ROOT**: "jumps" (Động từ chính).
- **Chủ ngữ (nsubj)**: "fox" (phụ thuộc vào "jumps").
- **Tân ngữ giới từ (pobj)**: "dog" (phụ thuộc vào giới từ "over", "over" phụ thuộc vào "jumps").
- **Bổ nghĩa cho "fox"**: "The" (det), "quick" (amod), "brown" (amod).

### 3.2 Trích xuất SVO
- **Input**: "The cat chased the mouse and the dog watched them."
- **Output**: `Found Triplet: (cat, chased, mouse)`
- **Nhận xét**: Code đã tách đúng mệnh đề đầu tiên. Tuy nhiên, mệnh đề sau "dog watched them" có thể không được bắt trọn vẹn nếu "them" không phải là `dobj` (tùy thuộc vào cách spaCy gán nhãn, đôi khi đại từ là `dobj` hoặc loại khác).

### 3.3 Noun Chunks: Thủ công vs spaCy
- **Input**: "The quick brown fox jumps over the lazy dog."
- **Manual**: `['The quick brown fox', 'the lazy dog']`
- **SpaCy**: `['The quick brown fox', 'the lazy dog']`
- **Nhận xét**:
  - Cách làm thủ công dựa trên quy tắc (`det`, `amod`, `compound`) cho kết quả tương đồng với spaCy trong trường hợp đơn giản.
  - Tuy nhiên, `doc.noun_chunks` của spaCy mạnh mẽ hơn vì nó xử lý được nhiều trường hợp phức tạp (như mệnh đề quan hệ rút gọn) mà quy tắc if-else đơn giản khó bao quát hết.

### 3.4 Đường đi đến gốc (Path to Root)
- **Input**: Token "brown" trong câu "The quick brown fox jumps over the lazy dog."
- **Output**: `brown -> fox -> jumps`
- **Phân tích**:
  - "brown" bổ nghĩa cho "fox" (quan hệ `amod`).
  - "fox" là chủ ngữ của "jumps" (quan hệ `nsubj`).
  - "jumps" là gốc của câu (`ROOT`).
  - Đường đi này giúp xác định vai trò và vị trí của từ trong cấu trúc tổng thể của câu.

---

## 4. Thách Thức và Giải Pháp

### 4.1 Thách Thức
- **Cấu trúc câu phức tạp**: Các câu bị động, câu hỏi, hoặc câu ghép phức hợp có thể làm sai lệch các quy tắc trích xuất đơn giản (ví dụ: chủ ngữ không phải lúc nào cũng là `nsubj`, có thể là `nsubjpass`).
- **Nhãn phụ thuộc đa dạng**: Hệ thống nhãn Universal Dependencies khá lớn và chi tiết, cần thời gian để nắm bắt hết ý nghĩa.
- **Hiệu năng**: Mô hình `en_core_web_md` nặng hơn `sm`, tuy chính xác hơn nhưng tốn tài nguyên hơn.

### 4.2 Giải Pháp
- **Sử dụng `spacy.explain()`**: Để tra cứu ý nghĩa của các nhãn (ví dụ: `spacy.explain("nsubj")`).
- **Mở rộng quy tắc**: Khi trích xuất thông tin, cần xét thêm các trường hợp như `nsubjpass` (chủ ngữ bị động), `agent` (tác nhân trong câu bị động).
- **Pattern Matching**: Sử dụng `spacy.matcher.Matcher` để định nghĩa các mẫu cú pháp phức tạp thay vì viết nhiều vòng lặp if-else lồng nhau.

---

## 5. Hướng Phát Triển
- **Trích xuất Quan hệ (Relation Extraction)**: Kết hợp Dependency Parsing với NER để trích xuất quan hệ giữa các thực thể (ví dụ: "Steve Jobs" --(founder of)--> "Apple").
- **Tóm tắt văn bản**: Sử dụng cây phụ thuộc để xác định các thành phần cốt lõi của câu (S-V-O) và loại bỏ các thành phần phụ để rút gọn câu.
- **Phân tích cảm xúc dựa trên khía cạnh (Aspect-based Sentiment Analysis)**: Dùng Dependency Parsing để liên kết từ chỉ cảm xúc (ví dụ: "good", "bad") với đối tượng cụ thể (ví dụ: "food", "service") trong câu "The food was good but the service was bad".
