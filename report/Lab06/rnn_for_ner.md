# Báo Cáo Lab 06: Nhận Dạng Thực Thể Tên (NER) với RNN (lab6_rnn_for_ner.ipynb)

## 📋 Mục Lục
1. [Giải Thích Các Bước Triển Khai](#1-giải-thích-các-bước-triển-khai)
2. [Hướng Dẫn Thực Thi Mã](#2-hướng-dẫn-thực-thi-mã)
3. [Phân Tích Kết Quả](#3-phân-tích-kết-quả)
4. [Thách Thức và Giải Pháp](#4-thách-thức-và-giải-pháp)
5. [Hướng Phát Triển](#5-hướng-phát-triển)

---

## 1. Giải Thích Các Bước Triển Khai

### Task 1: Tải và Tiền xử lý Dữ liệu
- **Mục đích**: Tải bộ dữ liệu chuẩn CoNLL-2003 và chuẩn bị từ điển.
- **Bước thực hiện**:
  - Sử dụng thư viện `datasets` của Hugging Face để tải `conll2003`.
  - Trích xuất câu (tokens) và nhãn (ner_tags).
  - Chuyển đổi nhãn từ dạng số sang dạng chuỗi (ví dụ: `0` -> `O`, `1` -> `B-PER`) để dễ kiểm soát.
  - Xây dựng `word_to_ix` (ánh xạ từ -> index) và `tag_to_ix` (ánh xạ nhãn -> index).
  - Thêm token đặc biệt: `<PAD>` (đệm), `<UNK>` (từ lạ).

### Task 2: Tạo PyTorch Dataset và DataLoader
- **Mục đích**: Đóng gói dữ liệu để huấn luyện theo batch.
- **Bước thực hiện**:
  - Tạo class `NERDataset` kế thừa `torch.utils.data.Dataset`.
  - Viết hàm `collate_fn` sử dụng `pad_sequence` để đệm các câu và chuỗi nhãn về cùng độ dài trong một batch.
  - Trả về thêm `lengths` (độ dài thực của câu) để sử dụng cho cơ chế `pack_padded_sequence`.

### Task 3: Xây dựng Mô hình RNN
- **Mục đích**: Xây dựng mô hình sequence labeling mạnh mẽ hơn POS Tagging.
- **Kiến trúc mô hình**:
  1. **Embedding Layer**: Chuyển đổi index từ sang vector (dim=100).
  2. **Bi-LSTM Layer**: Sử dụng **LSTM hai chiều** (Bidirectional LSTM) thay vì RNN thường. LSTM giúp giải quyết vấn đề vanishing gradient tốt hơn và nắm bắt phụ thuộc xa.
     - Hidden dim: 256.
     - Bidirectional: True (Output dim = 256 * 2 = 512).
  3. **Dropout**: Tỷ lệ 0.3 để giảm overfitting.
  4. **Linear Layer**: Ánh xạ output của LSTM sang số lượng nhãn NER.
- **Kỹ thuật**: Sử dụng `pack_padded_sequence` để tối ưu hóa tính toán, bỏ qua các token padding.

### Task 4: Huấn luyện Mô hình
- **Cấu hình**:
  - Optimizer: Adam (lr=0.001).
  - Loss function: CrossEntropyLoss (ignore_index=PAD_TAG).
  - Epochs: 10.
- **Quy trình**:
  - Tính toán Loss và Accuracy trên tập train sau mỗi epoch.
  - Accuracy được tính bằng cách so sánh nhãn dự đoán và nhãn thật, **bỏ qua các vị trí padding**.

### Task 5: Đánh giá Mô hình
- **Mục đích**: Kiểm tra khả năng tổng quát hóa trên tập Validation và Test.
- **Phương pháp**:
  - Sử dụng hàm `evaluate` để tính Loss và Accuracy trên tập Val/Test.
  - Đảm bảo không tính toán gradient (`torch.no_grad()`) để tiết kiệm bộ nhớ.
  - Viết hàm `predict_sentence` để dự đoán thực thể cho câu nhập vào bất kỳ.

---

## 2. Hướng Dẫn Thực Thi Mã

### 2.1 Yêu Cầu Hệ Thống
```bash
pip install torch datasets numpy
```

### 2.2 Cấu Trúc Thư Mục
```
 notebook/
    Lab06/
       lab6_rnn_for_ner.ipynb
```

### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook notebook/Lab06/lab6_rnn_for_ner.ipynb
   ```
2. **Chạy tuần tự các cell**:
   - Cell 1-2: Import thư viện và setup seed.
   - Cell 3-6: Tải dữ liệu CoNLL-2003 và xây dựng vocab.
   - Cell 7-9: Tạo Dataset và DataLoader.
   - Cell 10-11: Định nghĩa mô hình Bi-LSTM.
   - Cell 12-13: Huấn luyện mô hình (10 epochs).
   - Cell 14-15: Đánh giá trên tập Validation và Test.
   - Cell 16-17: Dự đoán câu mới.

---

## 3. Phân Tích Kết Quả

### 3.1 Quá Trình Huấn Luyện Chi Tiết
Dưới đây là kết quả chi tiết của quá trình huấn luyện qua 10 epochs:

| Epoch | Train Loss | Train Accuracy | Nhận Xét |
|:-----:|:----------:|:--------------:|:---------|
| 1     | 0.819      | 82.22%         | Khởi đầu tốt, mô hình học nhanh các quy luật cơ bản. |
| 2     | 0.521      | 86.07%         | Loss giảm mạnh, accuracy tăng gần 4%. |
| 3     | 0.399      | 88.46%         | |
| 4     | 0.316      | 90.67%         | Vượt mốc 90% accuracy. |
| 5     | 0.260      | 92.21%         | |
| 6     | 0.219      | 93.36%         | |
| 7     | 0.186      | 94.39%         | |
| 8     | 0.161      | 95.04%         | Đạt mốc 95% accuracy. |
| 9     | 0.140      | 95.68%         | |
| 10    | 0.121      | 96.21%         | Train loss rất thấp, mô hình học rất tốt trên tập train. |

### 3.2 Kết Quả Đánh Giá
- **Validation Accuracy**: **94.45%** (Loss: 0.242)
- **Test Accuracy**: **92.60%** (Loss: 0.349)

### 3.3 Nhận Xét

#### 1. Hiệu Suất Tổng Thể
- **Độ chính xác cao**: Mô hình đạt độ chính xác **92.60%** trên tập Test. Đây là kết quả rất ấn tượng cho bài toán NER, đặc biệt khi chỉ sử dụng kiến trúc Bi-LSTM cơ bản mà không có CRF hay pre-trained embeddings phức tạp (như BERT).
- **Khả năng tổng quát hóa**: Sự chênh lệch giữa Train Acc (96.21%) và Test Acc (92.60%) là khoảng 3.6%. Điều này cho thấy mô hình có hiện tượng overfitting nhẹ nhưng vẫn giữ được khả năng tổng quát hóa tốt trên dữ liệu chưa từng gặp.

#### 2. Vai Trò Của Bi-LSTM
- Việc sử dụng **LSTM hai chiều** là yếu tố then chốt. Trong NER, việc xác định một từ là thực thể hay không phụ thuộc rất nhiều vào ngữ cảnh cả hai phía.
- Ví dụ: Trong câu "Washington is a beautiful city", "Washington" là địa danh (LOC). Nhưng trong "Washington announced a new policy", "Washington" có thể là tổ chức (ORG) hoặc người (PER). Bi-LSTM giúp mô hình nhìn thấy từ "city" hoặc "announced" để đưa ra quyết định đúng.

### 3.4 Ví Dụ Dự Đoán
Câu: *"VNU University is located in Hanoi"*
- **Dự đoán**:
  - VNU: **B-ORG** (Tổ chức)
  - University: **I-ORG** (Tổ chức)
  - is: **O**
  - located: **O**
  - in: **O**
  - Hanoi: **O** (Dự đoán sai, lẽ ra phải là B-LOC)
- **Phân tích**:
  - Mô hình nhận diện đúng cụm "VNU University" là tổ chức (ORG).
  - Tuy nhiên, từ "Hanoi" bị dự đoán nhầm thành "O" (không phải thực thể). Điều này có thể do từ "Hanoi" ít xuất hiện trong tập train (CoNLL-2003 chủ yếu là dữ liệu tin tức phương Tây) hoặc do mô hình chưa đủ mạnh để bắt được ngữ cảnh này. Đây là minh chứng cho thách thức về **OOV (Out-of-Vocabulary)** và **Domain Adaptation**.

---

## 4. Thách Thức và Giải Pháp

### 4.1 Thách Thức
- **Dữ liệu mất cân bằng**: Nhãn `O` chiếm đa số áp đảo. Nếu không xử lý tốt, mô hình sẽ có xu hướng dự đoán mọi thứ là `O` để đạt accuracy cao (nhưng F1-score cho thực thể sẽ rất thấp).
- **Từ lạ (OOV)**: Tên riêng (như "Hanoi", "VNU") thường không có trong từ điển huấn luyện, dẫn đến việc mô hình phải dựa hoàn toàn vào ngữ cảnh hoặc gán token `<UNK>`.
- **Overfitting**: Với mô hình mạnh như LSTM và tập dữ liệu nhỏ/trung bình, mô hình dễ học thuộc lòng.

### 4.2 Giải Pháp
- **Masking**: Sử dụng `ignore_index` trong Loss function và masking khi tính Accuracy để loại bỏ ảnh hưởng của padding, giúp đánh giá chính xác hơn.
- **Dropout**: Sử dụng Dropout với tỷ lệ 0.3 để giảm thiểu overfitting.
- **Bi-LSTM**: Tận dụng ngữ cảnh toàn cục để giảm bớt sự phụ thuộc vào việc từ đó có trong từ điển hay không.

---

## 5. Hướng Phát Triển

### 5.1 Cải Tiến Mô Hình
- **Bi-LSTM-CRF**: Thêm lớp **CRF (Conditional Random Fields)**. CRF cực kỳ hữu ích trong NER vì nó học được các quy luật chuyển đổi nhãn (ví dụ: `I-ORG` không bao giờ đi sau `B-PER`). Điều này sẽ giúp sửa các lỗi dự đoán vô lý.
- **Pre-trained Embeddings**: Sử dụng **GloVe** hoặc **FastText** để khởi tạo embedding. FastText đặc biệt tốt cho NER vì nó sử dụng subword information, giúp xử lý tốt hơn các từ OOV và tên riêng.
- **Character-level Embedding**: Kết hợp thêm CNN/LSTM ở mức ký tự để mô hình học được các đặc điểm hình thái (như viết hoa, đuôi "-tion", "-ing").

### 5.2 Transformer & LLMs
- Chuyển sang sử dụng **BERT** (Bidirectional Encoder Representations from Transformers). BERT đã được pre-train trên lượng dữ liệu khổng lồ và hiểu ngữ cảnh sâu sắc hơn nhiều so với LSTM. Fine-tuning BERT trên CoNLL-2003 thường cho kết quả SOTA (>93-94% F1).

---
**Kết luận**: Bài thực hành đã xây dựng thành công hệ thống NER sử dụng Bi-LSTM với độ chính xác khả quan (~92.6% trên tập Test). Mặc dù còn một số hạn chế với các từ hiếm (như ví dụ "Hanoi"), nhưng đây là nền tảng vững chắc để phát triển các hệ thống trích xuất thông tin phức tạp hơn.
