# Báo Cáo Tổng Hợp Lab 06: Sequence Modeling với RNNs

## 📋 Mục Lục
1. [Part 1: Tìm hiểu về RNNs và Phân loại Token](#part-1-tìm-hiểu-về-rnns-và-phân-loại-token-lab6ipynb)
2. [Part 2: RNNs cho Phân loại Văn bản](#part-2-rnns-cho-phân-loại-văn-bản-lab6_rnns_text_classificationipynb)
3. [Part 3: Part-of-Speech Tagging với RNN](#part-3-part-of-speech-tagging-với-rnn-lab6_rnn_for_pos_taggingipynb)
4. [Part 4: Named Entity Recognition với RNN](#part-4-named-entity-recognition-với-rnn-lab6_rnn_for_neripynb)

---

## Part 1: Tìm hiểu về RNNs và Phân loại Token (lab6.ipynb)
### 1. Giải Thích Các Bước Triển Khai
#### Task 1: Làm quen với Tensor
- **Mục đích**: Hiểu cách tạo và thao tác với cấu trúc dữ liệu cơ bản của PyTorch.
- **Thực hiện**:
  - Tạo tensor từ list, numpy array.
  - Các phép toán cơ bản: cộng, nhân vô hướng, nhân ma trận (`@`).
  - Indexing, Slicing và Reshaping (`view`).

#### Task 2: Cơ chế Autograd
- **Mục đích**: Hiểu cách PyTorch tự động tính đạo hàm cho quá trình huấn luyện (Backpropagation).
- **Thực hiện**:
  - Tạo tensor với `requires_grad=True`.
  - Thực hiện chuỗi tính toán và gọi `.backward()` để tính gradient.
  - **Lưu ý**: Đồ thị tính toán được giải phóng sau khi gọi backward, nên không thể gọi lần 2 nếu không giữ lại graph.

#### Task 3: Các lớp Neural Network cơ bản
- **nn.Linear**: Lớp kết nối đầy đủ (Fully Connected), thực hiện phép biến đổi tuyến tính $y = xA^T + b$.
- **nn.Embedding**: Lớp quan trọng nhất cho NLP, chuyển đổi index của từ (số nguyên) thành vector dày (dense vector).
- **Xây dựng nn.Module**: Tạo class `MyFirstModel` kết hợp Embedding -> Linear -> Activation -> Linear.

### 2. Hướng Dẫn Thực Thi Mã
1. **Cài đặt**: `pip install torch numpy`
2. **Chạy**: Mở `lab6.ipynb` và chạy tuần tự các cell.

### 3. Phân Tích Kết Quả
- Đã thực hiện thành công các thao tác tensor và tính toán đạo hàm.
- Mô hình `MyFirstModel` chạy thành công forward pass, chuyển đổi input index thành output vector.
- Đây là bước đệm quan trọng để hiểu cách dữ liệu (câu chữ) được đưa vào mô hình Deep Learning.

### 4. Khó Khăn và Giải Pháp
- **Khái niệm Autograd**: Ban đầu khó hiểu về việc tại sao không thể gọi backward nhiều lần. -> **Giải pháp**: Đọc tài liệu về Dynamic Computation Graph của PyTorch.
- **Shape của Tensor**: Dễ nhầm lẫn khi nhân ma trận. -> **Giải pháp**: Luôn in `.shape` để kiểm tra.

---

## Part 2: RNNs cho Phân loại Văn bản (lab6_rnns_text_classification.ipynb)

### 1. Giải Thích Các Bước Triển Khai
#### Task 0: Data Loading & Label Encoding
- **Mục đích**: Chuẩn bị dữ liệu cho quá trình huấn luyện.
- **Bước thực hiện**:
  - Đọc dữ liệu từ các file CSV (train.csv, val.csv, test.csv) trong folder hwu.
  - Sử dụng `LabelEncoder` để chuyển đổi nhãn "category" từ dạng text sang số.
  - Kiểm tra kích thước và cấu trúc dữ liệu.

#### Task 1: TF-IDF + Logistic Regression
- **Mục đích**: Xây dựng baseline model sử dụng phương pháp truyền thống.
- **Bước thực hiện**:
  - Sử dụng `TfidfVectorizer` để chuyển văn bản thành vector.
  - Kết hợp với `LogisticRegression` trong một pipeline.
  - Huấn luyện trên tập train và đánh giá trên tập test.

#### Task 2: Word2Vec (Average) + Dense Layer
- **Mục đích**: Sử dụng word embeddings nhưng chưa xử lý được tính tuần tự của văn bản.
- **Bước thực hiện**:
  - Huấn luyện mô hình Word2Vec trên dữ liệu training.
  - Chuyển đổi mỗi câu thành vector trung bình của các từ.
  - Xây dựng mạng neural với Dense layers để phân loại.

#### Task 3: Pre-trained Embedding + LSTM
- **Mục đích**: Sử dụng LSTM với embedding đã được pre-train từ Word2Vec.
- **Bước thực hiện**:
  - Tokenize văn bản và padding sequences.
  - Tạo embedding matrix từ Word2Vec đã huấn luyện.
  - Xây dựng mô hình Sequential: Embedding (frozen) → LSTM → Dense.
  - Sử dụng EarlyStopping để tránh overfitting.

#### Task 4: Scratch Embedding + LSTM
- **Mục đích**: So sánh hiệu quả khi embedding layer được học từ đầu.
- **Bước thực hiện**:
  - Sử dụng cùng architecture như Task 3 nhưng embedding layer trainable.
  - Huấn luyện end-to-end với EarlyStopping.

#### Task 5: Evaluation & Analysis
- **Mục đích**: So sánh định lượng và định tính giữa các mô hình.
- **Bước thực hiện**:
  - Tính macro F1-score và test loss cho cả 4 mô hình.
  - Phân tích qualitative trên các câu khó có cấu trúc phủ định/phức tạp.
  - Tạo bảng so sánh và nhận xét.

### 2. Hướng Dẫn Thực Thi Mã
#### 2.1 Yêu Cầu Hệ Thống
```bash
pip install pandas numpy scikit-learn gensim tensorflow jupyter
```

#### 2.2 Cấu Trúc Thư Mục
```
Lab06/
├── lab6_rnns_text_classification.ipynb
├── hwu/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── rnns_text_classification.md
```

#### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook lab6_rnns_text_classification.ipynb
   ```

2. **Chạy từng cell theo thứ tự**:
   - Cell 1: Import libraries và setup random seed.
   - Cell 2: Load dữ liệu từ CSV files.
   - Cell 3: Label encoding.
   - Cell 4-5: Task 1 (TF-IDF + LR).
   - Cell 6-7: Task 2 (Word2Vec + Dense).
   - Cell 8-9: Task 3 (Pre-trained Embedding + LSTM).
   - Cell 10-11: Task 4 (Scratch Embedding + LSTM).
   - Cell 12-13: Task 5 (Evaluation & Analysis).

### 3. Phân Tích Kết Quả
#### 3.1 Kết quả Task 1
```
                accuracy                           0.84      1076
               macro avg       0.85      0.83      0.84      1076
            weighted avg       0.84      0.84      0.84      1076
```
- Accuracy 0.84 : Đây là mức tốt cho baseline TF-IDF + LR, nhất là nếu dữ liệu có nhiều nhãn hoặc câu ngắn.
- Macro avg ~ Weighted avg : Cho thấy các lớp được cân bằng tốt, không có lớp nào bị bỏ quên.

#### 3.2 Kết quả Task 2
```
                accuracy                           0.35      1076
               macro avg       0.34      0.33      0.30      1076
            weighted avg       0.35      0.35      0.32      1076
```
- Accuracy 0.35 : Thấp hơn nhiều so với TF-IDF + LR, cho thấy việc mất thông tin về thứ tự từ ảnh hưởng lớn.
- Tập train nhỏ có thể không đủ để học embeddings tốt.
- Vector trung bình làm mất ngữ cảnh.
- Mạng nông không đủ mạnh để bù đắp.

#### 3.3 Kết quả Task 3
```
                accuracy                           0.40      1076
               macro avg       0.39      0.39      0.38      1076
            weighted avg       0.40      0.40      0.39      1076
```
- Accuracy 0.40 : Cải thiện so với Task 2, cho thấy LSTM giúp nắm bắt thông tin tuần tự.
- Pre-trained embeddings giúp mô hình học nhanh hơn và hiệu quả hơn.
- Tuy nhiên, vẫn chưa vượt trội so với TF-IDF + LR có thể do tập dữ liệu nhỏ và cấu trúc câu đơn giản.

#### 3.4 Kết quả Task 4
```
                accuracy                           0.27      1076
               macro avg       0.16      0.25      0.18      1076
            weighted avg       0.17      0.27      0.20      1076
```
- Accuracy 0.27 : Thấp hơn so với Task 3, cho thấy việc học embedding từ đầu gặp khó khăn với tập dữ liệu nhỏ.
- Sử dụng early stopping giúp tránh overfitting nhưng mô hình vẫn chưa học được biểu diễn tốt.
- Cần nhiều dữ liệu hơn để embedding layer học hiệu quả.

#### 3.5 Kết quả Task 5
- **Bảng Tổng Hợp Kết Quả**
```
| Pipeline | F1-score (Macro) | Test Loss | Nhận Xét |
|----------|------------------|-----------|----------|
| TF-IDF + Logistic Regression | 0.835298 | 1.050197 | Hiệu quả bất ngờ, đơn giản và nhanh. |
| Word2Vec (Avg) + Dense | 0.304154 | 2.452722 | Mất thông tin thứ tự, mạng nông. |
| Pre-trained Embedding + LSTM | 0.376478 | 2.108491 | Xử lý tuần tự ổn, tuy nhiên cần tuning thêm. |
| Scratch Embedding + LSTM | 0.178246 | 2.868297 | Flexible nhưng cần nhiều dữ liệu hơn. |
```

- Nhận Xét Chung:
  - TF-IDF + LR vẫn là lựa chọn tốt cho tập dữ liệu nhỏ và câu đơn giản.
  - Mô hình dựa trên embeddings và LSTM cần nhiều dữ liệu hơn để phát huy hiệu quả.
  - Pre-trained embeddings giúp cải thiện so với học từ đầu, nhưng vẫn chưa đủ để vượt qua phương pháp truyền thống.

- **Phân Tích Định Tính Các Câu Khó**
```
| Sentence | True Intent | TF-IDF + LR | W2V Avg + Dense | Pretrained LSTM | Scratch LSTM |
|-----------|--------------|--------------|------------------|------------------|---------------|
| can you remind me to not call my mom | reminder_create | calendar_set | general_quirky | takeaway_query | email_sendemail |
| is it going to be sunny or rainy tomorrow | weather_query | weather_query | qa_maths | qa_maths | takeaway_order |
| find a flight from new york to london but not ... | flight_search | general_negate | transport_query | email_sendemail | calendar_set |
```

- **Câu 1:** "can you remind me to not call my mom" với phủ định "not call"
  - TF-IDF không hiểu phủ định dễ nhầm “remind” với “calendar_set”
  - W2V Avg mất ngữ cảnh, chọn intent chung chung.
  - Pre-trained LSTM Nắm bắt phủ định tốt hơn, nhưng vẫn nhầm với “takeaway_query”.
  - Scratch LSTM Không học được biểu diễn ngữ nghĩa, chủ yếu dựa vào từ khóa.

- **Câu 2:** "is it going to be sunny or rainy tomorrow"
  - TF-IDF + LR đúng intent nhờ từ khóa “sunny”, “rainy”.
  - W2V Avg + Dense không hiểu ngữ cảnh, chọn intent không liên quan.
  - Pre-trained LSTM và Scratch LSTM chưa nắm bắt được ý định chính xác, dễ nhầm lẫn giữa các lớp hỏi đáp.

- **Câu 3:** "find a flight from new york to london but not ..."
  - TF-IDF + LR không hiểu ngữ cảnh “find a flight”, nhầm sang intent “general_negate”.
  - W2V Avg + Dense không nắm bắt được ý định chính xác.
  - Pre-trained LSTM và Scratch LSTM đều không hiểu phủ định “but not”, dẫn đến nhầm lẫn.

### 4. Thách Thức và Giải Pháp
#### 4.1 Thách Thức
- Dữ liệu nhỏ hạn chế khả năng học biểu diễn tốt.
- Các câu phức tạp với phủ định, mệnh đề đa nghĩa.
- Overfitting do mô hình phức tạp với dữ liệu hạn chế.

#### 4.2 Giải Pháp
- Sử dụng pre-trained embeddings để tận dụng kiến thức ngôn ngữ có sẵn.
- Fine tuning mô hình ngữ cảnh.
- Kết hợp nhiều đặc trưng (TF-IDF + embeddings) hoặc (TF-IDF + LSTM) để tận dụng ưu điểm của cả hai.
- Data augmentation để tăng kích thước tập huấn luyện.
- Regularization techniques như Dropout, Early Stopping để giảm overfitting.
- Hyperparameter tuning để tìm cấu hình tối ưu.


### 5 Hướng Phát Triển
- Thử nghiệm với Transformer models (BERT, DistilBERT).
- Ensemble methods kết hợp multiple approaches.
- Advanced preprocessing (spelling correction, normalization).
- Hyperparameter optimization với tools như Optuna.

---

## Part 3: Part-of-Speech Tagging với RNN (lab6_rnn_for_pos_tagging.ipynb)

### 1. Giải Thích Các Bước Triển Khai
#### Task 1: Tải và Tiền xử lý Dữ liệu
- **Mục đích**: Đọc dữ liệu từ định dạng CoNLL-U và xây dựng từ điển.
- **Bước thực hiện**:
  - Đọc file `.conllu` (train/dev) để lấy danh sách các câu, mỗi câu gồm các cặp `(word, upos)`.
  - Xây dựng `word_to_ix` (ánh xạ từ -> index) và `tag_to_ix` (ánh xạ nhãn -> index).
  - Thêm các token đặc biệt: `<PAD>` (đệm), `<UNK>` (từ lạ).
  - **Kết quả**:
    - Train sentences: 12,544 câu.
    - Dev sentences: 2,001 câu.
    - Vocab size: 6,733 từ (min_freq=3).
    - Tag size: 18 nhãn (bao gồm `<PAD>`).

#### Task 2: Tạo PyTorch Dataset và DataLoader
- **Mục đích**: Chuẩn bị dữ liệu dạng batch để huấn luyện mô hình.
- **Bước thực hiện**:
  - Tạo class `POSDataset` kế thừa từ `torch.utils.data.Dataset`.
  - Viết hàm `collate_fn` sử dụng `pad_sequence` để đệm các câu trong batch về cùng độ dài.
  - Tạo `DataLoader` cho tập train (shuffle=True) và dev (shuffle=False) với `batch_size=64`.

#### Task 3: Xây dựng Mô hình RNN
- **Mục đích**: Xây dựng mô hình sequence labeling sử dụng RNN.
- **Kiến trúc mô hình**:
  1. **Embedding Layer**: Chuyển đổi index của từ thành vector (dim=100).
  2. **RNN Layer**: Bidirectional RNN (hidden_dim=128) để nắm bắt ngữ cảnh hai chiều.
  3. **Dropout**: Giảm overfitting (p=0.1).
  4. **Linear Layer**: Ánh xạ output của RNN sang không gian nhãn (output_dim=18).
- **Kỹ thuật đặc biệt**: Sử dụng `pack_padded_sequence` và `pad_packed_sequence` để RNN bỏ qua các token padding, tăng hiệu quả tính toán.

#### Task 4: Huấn luyện Mô hình
- **Cấu hình**:
  - Optimizer: Adam (lr=1e-3).
  - Loss function: CrossEntropyLoss (ignore_index=PAD_TAG).
  - Epochs: 20.
- **Quy trình**:
  - Forward pass -> Tính loss -> Backward pass -> Update weights.
  - Theo dõi loss và accuracy trên tập dev sau mỗi epoch.
  - Lưu lại trạng thái mô hình tốt nhất (best dev accuracy).

#### Task 5: Đánh giá Mô hình
- **Mục đích**: Kiểm tra hiệu suất mô hình trên tập dev và dự đoán câu mới.
- **Phương pháp**:
  - Tính accuracy trên các token thực (bỏ qua padding).
  - Viết hàm `predict_sentence` để gán nhãn cho câu nhập vào bất kỳ.

### 2. Hướng Dẫn Thực Thi Mã
#### 2.1 Yêu Cầu Hệ Thống
```bash
pip install torch numpy
```

#### 2.2 Cấu Trúc Thư Mục
```
Lab06/
├── lab6_rnn_for_pos_tagging.ipynb
├── rnn_for_pos_tagging.md
└── ../UD_English-EWT/
    ├── en_ewt-ud-train.conllu
    └── en_ewt-ud-dev.conllu
```

#### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook lab6_rnn_for_pos_tagging.ipynb
   ```
2. **Chạy tuần tự các cell**:
   - Cell 1-2: Import và setup.
   - Cell 3-5: Task 1 (Data Loading & Vocab).
   - Cell 6-8: Task 2 (Dataset & DataLoader).
   - Cell 9-10: Task 3 (Model Definition).
   - Cell 11-13: Task 4 (Training).
   - Cell 14-15: Task 5 (Evaluation & Demo).

### 3. Phân Tích Kết Quả

#### 3.1 Quá Trình Huấn Luyện Chi Tiết
Dưới đây là kết quả chi tiết của quá trình huấn luyện qua 20 epochs:

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Nhận Xét |
|:-----:|:----------:|:--------:|:------------:|:---------|
| 1     | 67.9458    | 33.3044  | 0.7603       | Khởi đầu tốt, mô hình học nhanh các quy luật cơ bản. |
| 2     | 36.7162    | 25.0620  | 0.8155       | Loss giảm mạnh (~50%), accuracy tăng >5%. |
| 3     | 27.3685    | 20.9647  | 0.8466       | |
| 4     | 21.4835    | 18.4810  | 0.8632       | |
| 5     | 17.7275    | 16.6868  | 0.8767       | Kết thúc giai đoạn học nhanh. |
| 6     | 15.0122    | 15.5063  | 0.8876       | |
| 7     | 12.7403    | 14.9340  | 0.8937       | Dev loss bắt đầu ổn định quanh mức 14-15. |
| 8     | 11.3818    | 14.6995  | 0.8957       | |
| 9     | 10.1980    | 14.7442  | 0.8952       | |
| 10    | 9.1291     | 15.0488  | 0.8945       | |
| 11    | 8.2330     | 14.5638  | 0.8999       | Tiệm cận mức 90%. |
| 12    | 7.4461     | 14.9955  | 0.8995       | |
| 13    | 6.8169     | 14.9773  | 0.9035       | Đạt độ chính xác cao nhất (Best Model). |
| 14    | 6.2771     | 15.7560  | 0.8972       | Dev loss bắt đầu tăng -> Dấu hiệu Overfitting. |
| 15    | 5.5244     | 16.7355  | 0.8951       | |
| 16    | 5.0551     | 16.2119  | 0.9021       | |
| 17    | 4.5238     | 16.8760  | 0.9006       | |
| 18    | 3.9932     | 17.4442  | 0.9014       | |
| 19    | 3.6125     | 18.6773  | 0.8960       | |
| 20    | 3.2173     | 18.6980  | 0.9000       | Train loss rất thấp nhưng Dev loss cao nhất. |

#### 3.2 Nhận Xét

##### 1. Hiệu Suất Tổng Thể
- **Đỉnh cao (Peak Performance)**: Mô hình đạt độ chính xác tốt nhất là **90.35%** tại Epoch 13. Đây là kết quả rất khả quan cho một mô hình RNN đơn giản (không dùng pre-trained embeddings hay kiến trúc phức tạp như Transformer).
- **Tốc độ hội tụ**: Mô hình hội tụ khá nhanh. Chỉ sau 5 epochs đầu tiên, độ chính xác đã đạt 87.67%. Các epochs sau đó chủ yếu tinh chỉnh các trường hợp khó (như từ đa nghĩa, từ hiếm).

##### 2. Phân Tích Overfitting
- **Giai đoạn 1 (Epoch 1-11)**: Cả Train Loss và Dev Loss đều giảm. Đây là giai đoạn "Learning" hiệu quả nhất.
- **Giai đoạn 2 (Epoch 12-13)**: Dev Loss đi ngang (khoảng 14.9), trong khi Train Loss tiếp tục giảm. Đây là điểm tối ưu ("Sweet Spot").
- **Giai đoạn 3 (Epoch 14-20)**:
    - **Train Loss** giảm sâu xuống 3.2 (mô hình học thuộc lòng dữ liệu huấn luyện).
    - **Dev Loss** tăng ngược lại lên 18.7 (mô hình mất khả năng tổng quát hóa).
    - **Kết luận**: Việc huấn luyện thêm sau epoch 13 không mang lại lợi ích về độ chính xác và làm giảm tính tổng quát của mô hình. Cơ chế **Early Stopping** nên được kích hoạt tại đây.

##### 3. Vai Trò Của Bidirectional RNN
- Việc accuracy đạt >90% chứng tỏ kiến trúc 2 chiều (Bidirectional) rất hiệu quả.
- Kết quả 90.35% cho thấy mô hình thực sự học được cấu trúc ngữ pháp chứ không chỉ nhớ vẹt.

#### 3.3 Ví Dụ Dự Đoán
Câu: *"I love NLP ."*
- **Dự đoán**: `[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'PROPN'), ('.', 'PUNCT')]`
- **Phân tích**:
  - "I" -> PRON (Đại từ): Chính xác.
  - "love" -> VERB (Động từ): Chính xác.
  - "NLP" -> PROPN (Danh từ riêng): Chính xác.
  - "." -> PUNCT (Dấu câu): Chính xác.

### 4. Thách Thức và Giải Pháp
#### 4.1 Thách Thức
- **Từ lạ (OOV - Out of Vocabulary)**: Các từ không xuất hiện trong tập train sẽ bị gán là `<UNK>`, làm giảm độ chính xác.
- **Từ đa nghĩa**: Một từ có thể có nhiều nhãn tùy ngữ cảnh (vd: "book" có thể là NOUN hoặc VERB).
- **Overfitting**: Sau khoảng 13 epochs, mô hình bắt đầu học thuộc lòng tập train (loss train giảm sâu nhưng loss dev tăng).

#### 4.2 Giải Pháp
- **Early Stopping**: Dừng huấn luyện khi accuracy trên tập dev không cải thiện sau một số epoch nhất định (trong trường hợp này là sau epoch 13).
- **Xử lý OOV**: Sử dụng token `<UNK>` và thay thế các từ tần suất thấp bằng `<UNK>` khi training để mô hình học cách xử lý từ lạ.
- **Ngữ cảnh**: Sử dụng **Bidirectional RNN** để xem xét ngữ cảnh toàn cục.
- **Padding & Packing**: Sử dụng `pad_sequence` kết hợp `pack_padded_sequence` để xử lý batch hiệu quả mà không tính toán trên phần đệm.

### 5. Hướng Phát Triển
#### 5.1 Cải Tiến Mô Hình
- **LSTM/GRU**: Thay thế RNN thường bằng LSTM hoặc GRU để xử lý phụ thuộc xa tốt hơn (tránh vanishing gradient).
- **CRF (Conditional Random Fields)**: Thêm lớp CRF lên trên RNN để mô hình hóa sự phụ thuộc giữa các nhãn liên tiếp (vd: ADJ thường đứng trước NOUN).
- **Pre-trained Embeddings**: Sử dụng GloVe hoặc Word2Vec thay vì học embedding từ đầu.

#### 5.2 Tối Ưu Hóa
- **Hyperparameter Tuning**: Thử nghiệm với learning rate, hidden dim, số layers khác nhau.
- **Data Augmentation**: Tăng cường dữ liệu để mô hình tổng quát hóa tốt hơn.

---

## Part 4: Named Entity Recognition với RNN (lab6_rnn_for_ner.ipynb)

### 1. Giải Thích Các Bước Triển Khai

#### Task 1: Tải và Tiền xử lý Dữ liệu
- **Mục đích**: Tải bộ dữ liệu chuẩn CoNLL-2003 và chuẩn bị từ điển.
- **Bước thực hiện**:
  - Sử dụng thư viện `datasets` của Hugging Face để tải `conll2003`.
  - Trích xuất câu (tokens) và nhãn (ner_tags).
  - Chuyển đổi nhãn từ dạng số sang dạng chuỗi (ví dụ: `0` -> `O`, `1` -> `B-PER`) để dễ kiểm soát.
  - Xây dựng `word_to_ix` (ánh xạ từ -> index) và `tag_to_ix` (ánh xạ nhãn -> index).
  - Thêm token đặc biệt: `<PAD>` (đệm), `<UNK>` (từ lạ).

#### Task 2: Tạo PyTorch Dataset và DataLoader
- **Mục đích**: Đóng gói dữ liệu để huấn luyện theo batch.
- **Bước thực hiện**:
  - Tạo class `NERDataset` kế thừa `torch.utils.data.Dataset`.
  - Viết hàm `collate_fn` sử dụng `pad_sequence` để đệm các câu và chuỗi nhãn về cùng độ dài trong một batch.
  - Trả về thêm `lengths` (độ dài thực của câu) để sử dụng cho cơ chế `pack_padded_sequence`.

#### Task 3: Xây dựng Mô hình RNN
- **Mục đích**: Xây dựng mô hình sequence labeling mạnh mẽ hơn POS Tagging.
- **Kiến trúc mô hình**:
  1. **Embedding Layer**: Chuyển đổi index từ sang vector (dim=100).
  2. **Bi-LSTM Layer**: Sử dụng **LSTM hai chiều** (Bidirectional LSTM) thay vì RNN thường. LSTM giúp giải quyết vấn đề vanishing gradient tốt hơn và nắm bắt phụ thuộc xa.
     - Hidden dim: 256.
     - Bidirectional: True (Output dim = 256 * 2 = 512).
  3. **Dropout**: Tỷ lệ 0.3 để giảm overfitting.
  4. **Linear Layer**: Ánh xạ output của LSTM sang số lượng nhãn NER.
- **Kỹ thuật**: Sử dụng `pack_padded_sequence` để tối ưu hóa tính toán, bỏ qua các token padding.

#### Task 4: Huấn luyện Mô hình
- **Cấu hình**:
  - Optimizer: Adam (lr=0.001).
  - Loss function: CrossEntropyLoss (ignore_index=PAD_TAG).
  - Epochs: 10.
- **Quy trình**:
  - Tính toán Loss và Accuracy trên tập train sau mỗi epoch.
  - Accuracy được tính bằng cách so sánh nhãn dự đoán và nhãn thật, **bỏ qua các vị trí padding**.

#### Task 5: Đánh giá Mô hình
- **Mục đích**: Kiểm tra khả năng tổng quát hóa trên tập Validation và Test.
- **Phương pháp**:
  - Sử dụng hàm `evaluate` để tính Loss và Accuracy trên tập Val/Test.
  - Đảm bảo không tính toán gradient (`torch.no_grad()`) để tiết kiệm bộ nhớ.
  - Viết hàm `predict_sentence` để dự đoán thực thể cho câu nhập vào bất kỳ.

### 2. Hướng Dẫn Thực Thi Mã

#### 2.1 Yêu Cầu Hệ Thống
```bash
pip install torch datasets numpy
```

#### 2.2 Cấu Trúc Thư Mục
```
Lab06/
├── lab6_rnn_for_ner.ipynb
├── rnn_for_ner.md
```

#### 2.3 Cách Chạy
1. **Mở Jupyter Notebook**:
   ```bash
   jupyter notebook lab6_rnn_for_ner.ipynb
   ```
2. **Chạy tuần tự các cell**:
   - Cell 1-2: Import thư viện và setup seed.
   - Cell 3-6: Tải dữ liệu CoNLL-2003 và xây dựng vocab.
   - Cell 7-9: Tạo Dataset và DataLoader.
   - Cell 10-11: Định nghĩa mô hình Bi-LSTM.
   - Cell 12-13: Huấn luyện mô hình (10 epochs).
   - Cell 14-15: Đánh giá trên tập Validation và Test.
   - Cell 16-17: Dự đoán câu mới.

### 3. Phân Tích Kết Quả

#### 3.1 Quá Trình Huấn Luyện Chi Tiết
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
| 10    | 0.121      | **96.21%**     | Train loss rất thấp, mô hình học rất tốt trên tập train. |

#### 3.2 Kết Quả Đánh Giá
- **Validation Accuracy**: **94.45%** (Loss: 0.242)
- **Test Accuracy**: **92.60%** (Loss: 0.349)

#### 3.3 Nhận Xét

##### 1. Hiệu Suất Tổng Thể
- **Độ chính xác cao**: Mô hình đạt độ chính xác **92.60%** trên tập Test. Đây là kết quả rất ấn tượng cho bài toán NER, đặc biệt khi chỉ sử dụng kiến trúc Bi-LSTM cơ bản mà không có CRF hay pre-trained embeddings phức tạp (như BERT).
- **Khả năng tổng quát hóa**: Sự chênh lệch giữa Train Acc (96.21%) và Test Acc (92.60%) là khoảng 3.6%. Điều này cho thấy mô hình có hiện tượng overfitting nhẹ nhưng vẫn giữ được khả năng tổng quát hóa tốt trên dữ liệu chưa từng gặp.

##### 2. Vai Trò Của Bi-LSTM
- Việc sử dụng **LSTM hai chiều** là yếu tố then chốt. Trong NER, việc xác định một từ là thực thể hay không phụ thuộc rất nhiều vào ngữ cảnh cả hai phía.
- Ví dụ: Trong câu "Washington is a beautiful city", "Washington" là địa danh (LOC). Nhưng trong "Washington announced a new policy", "Washington" có thể là tổ chức (ORG) hoặc người (PER). Bi-LSTM giúp mô hình nhìn thấy từ "city" hoặc "announced" để đưa ra quyết định đúng.

#### 3.4 Ví Dụ Dự Đoán
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

### 4. Thách Thức và Giải Pháp

#### 4.1 Thách Thức
- **Dữ liệu mất cân bằng**: Nhãn `O` chiếm đa số áp đảo. Nếu không xử lý tốt, mô hình sẽ có xu hướng dự đoán mọi thứ là `O` để đạt accuracy cao (nhưng F1-score cho thực thể sẽ rất thấp).
- **Từ lạ (OOV)**: Tên riêng (như "Hanoi", "VNU") thường không có trong từ điển huấn luyện, dẫn đến việc mô hình phải dựa hoàn toàn vào ngữ cảnh hoặc gán token `<UNK>`.
- **Overfitting**: Với mô hình mạnh như LSTM và tập dữ liệu nhỏ/trung bình, mô hình dễ học thuộc lòng.

#### 4.2 Giải Pháp
- **Masking**: Sử dụng `ignore_index` trong Loss function và masking khi tính Accuracy để loại bỏ ảnh hưởng của padding, giúp đánh giá chính xác hơn.
- **Dropout**: Sử dụng Dropout với tỷ lệ 0.3 để giảm thiểu overfitting.
- **Bi-LSTM**: Tận dụng ngữ cảnh toàn cục để giảm bớt sự phụ thuộc vào việc từ đó có trong từ điển hay không.

### 5. Hướng Phát Triển

#### 5.1 Cải Tiến Mô Hình
- **Bi-LSTM-CRF**: Thêm lớp **CRF (Conditional Random Fields)**. CRF cực kỳ hữu ích trong NER vì nó học được các quy luật chuyển đổi nhãn (ví dụ: `I-ORG` không bao giờ đi sau `B-PER`). Điều này sẽ giúp sửa các lỗi dự đoán vô lý.
- **Pre-trained Embeddings**: Sử dụng **GloVe** hoặc **FastText** để khởi tạo embedding. FastText đặc biệt tốt cho NER vì nó sử dụng subword information, giúp xử lý tốt hơn các từ OOV và tên riêng.
- **Character-level Embedding**: Kết hợp thêm CNN/LSTM ở mức ký tự để mô hình học được các đặc điểm hình thái (như viết hoa, đuôi "-tion", "-ing").

#### 5.2 Transformer & LLMs
- Chuyển sang sử dụng **BERT** (Bidirectional Encoder Representations from Transformers). BERT đã được pre-train trên lượng dữ liệu khổng lồ và hiểu ngữ cảnh sâu sắc hơn nhiều so với LSTM. Fine-tuning BERT trên CoNLL-2003 thường cho kết quả SOTA (>93-94% F1).

---
**Kết luận**: Bài thực hành đã xây dựng thành công hệ thống NER sử dụng Bi-LSTM với độ chính xác khả quan (~92.6% trên tập Test). Mặc dù còn một số hạn chế với các từ hiếm (như ví dụ "Hanoi"), nhưng đây là nền tảng vững chắc để phát triển các hệ thống trích xuất thông tin phức tạp hơn.
