# Báo Cáo Lab: Enhanced Spark NLP Pipeline với TF-IDF và Document Similarity

## Cấu trúc dự án
```
 src/
    Lab02/
       spark_labs/
          src/
             main/
                scala/
                   com/
                      HMina/
                         spark/
                            Lab17_NLPPipeline.scala   # Main Spark Application
 data/
    c4-train.00000-of-01024-30K.json                  # Dữ liệu huấn luyện
```

## Các Bước Triển Khai

### 1. Thiết Lập Môi Trường
- **Phiên bản Scala**: 2.12.18
- **Phiên bản Spark**: 3.5.1
- **Phiên bản Java**: 19.0.1
- **Phiên bản SBT**: 1.11.0

### 2. Enhanced Features (New Enhancements)
- **Customizable Document Limit**: Variable `limitDocuments` để dễ dàng thay đổi số lượng documents
- **Detailed Performance Measurement**: Đo thời gian thực thi từng stage chính
- **Vector Normalization**: Normalizer layer để chuẩn hóa TF-IDF vectors  
- **Document Similarity Analysis**: Tìm top 5 documents tương tự nhất sử dụng cosine similarity

### 3. Tải Dữ Liệu
```scala
val limitDocuments = 1000 // Easily configurable document limit
val df = spark.read.json("../../c4-train.00000-of-01024-30K.json")
  .limit(limitDocuments) // Use configurable limit
val textDF = df.select("text").na.drop()
```
- Tải thành công bộ dữ liệu C4 compressed (.json.gz) vào Spark DataFrame
- Spark tự động xử lý decompression cho file .gz
- Áp dụng làm sạch dữ liệu bằng cách loại bỏ giá trị null
- Sử dụng biến `limitDocuments` để dễ dàng điều chỉnh kích thước dataset

### 4. Dual Pipeline Implementation (Enhanced Feature)
```scala
// Pipeline 1: Basic TF-IDF (no normalization)
val basicPipeline = new Pipeline()
  .setStages(Array(tokenizer, remover, hashingTF, idf))

// Pipeline 2: Enhanced TF-IDF + Normalizer
val enhancedPipeline = new Pipeline()
  .setStages(Array(tokenizer, remover, hashingTF, idf, normalizer))
```

**Pipeline 1 - Basic TF-IDF Stages:**
1. **RegexTokenizer**: Phân tách văn bản sử dụng pattern "\\s+|[.,;!?()\"']" (khoảng trắng và dấu câu)
2. **StopWordsRemover**: Lọc bỏ các từ dừng tiếng Anh phổ biến
3. **HashingTF**: Chuyển đổi tokens thành vectors tần suất từ (20,000 features)
4. **IDF**: Áp dụng trọng số tần suất nghịch đảo văn bản

**Pipeline 2 - Enhanced TF-IDF + Normalization Stages:**
1. **RegexTokenizer**: Phân tách văn bản sử dụng pattern "\\s+|[.,;!?()\"']" (khoảng trắng và dấu câu)
2. **StopWordsRemover**: Lọc bỏ các từ dừng tiếng Anh phổ biến
3. **HashingTF**: Chuyển đổi tokens thành vectors tần suất từ (20,000 features)
4. **IDF**: Áp dụng trọng số tần suất nghịch đảo văn bản
5. **Normalizer** (New): Chuẩn hóa vectors với L2 normalization để tối ưu cosine similarity

### 5. Chiến Lược Tokenization
- **Chính**: RegexTokenizer với pattern "\\s+|[.,;!?()\"']"
- **Thay thế**: Basic Tokenizer (dựa trên khoảng trắng) có sẵn dưới dạng comment
- RegexTokenizer xử lý tốt hơn dấu câu và ký tự đặc biệt bằng cách tách theo khoảng trắng và các dấu câu phổ biến

### 6. Enhanced Vector Processing
- **HashingTF**: 
  - Không gian đặc trưng: 20,000 chiều
  - Xử lý va chạm hash cho từ vựng lớn
  - Sử dụng bộ nhớ hiệu quả
- **IDF**: 
  - Giảm tầm quan trọng của các từ xuất hiện thường xuyên
  - Nhấn mạnh các từ độc đáo, mang tính thông tin
- **Normalizer (New)**: 
  - L2 normalization để chuẩn hóa độ dài vector
  - Tối ưu hóa cho cosine similarity calculation
  - Đảm bảo so sánh documents dựa trên nội dung thay vì độ dài

### 7. Dual Pipeline Training & Comparison (New Feature)
```scala
// Train Basic Pipeline
val basicModel = basicPipeline.fit(textDF)
val basicResult = basicModel.transform(textDF)

// Train Enhanced Pipeline  
val enhancedModel = enhancedPipeline.fit(textDF)
val enhancedResult = enhancedModel.transform(textDF)
```
- **Training Comparison**: So sánh thời gian training giữa 2 pipelines
- **Performance Analysis**: Tính toán overhead của Normalizer
- **Result Comparison**: So sánh output vectors giữa basic và enhanced pipeline

### 8. Document Similarity Analysis (New Feature)
- **Cosine Similarity**: Tính toán độ tương tự giữa documents sử dụng normalized vectors
- **Top-K Search**: Tìm 5 documents tương tự nhất với reference document
- **Performance Optimized**: Sử dụng enhanced pipeline results để accuracy cao hơn
- **Normalized Vector Benefits**: Cosine similarity = dot product cho normalized vectors

## Cách Chạy Code
### Các Bước Thực Thi
```bash
# Di chuyển đến thư mục dự án
cd src/Lab02/spark_labs

# Tạo các thư mục cần thiết
mkdir log, results

# Biên dịch dự án
sbt compile

# Chạy ứng dụng
sbt run
```

### Cấu Trúc Kết Quả Enhanced
```
log/
├── pipeline_YYYYMMDD_HHMMSS.log    # Enhanced log với performance metrics
results/
├── lab17_pipeline_output.txt       # Kết quả với similarity analysis
```

### Enhanced Log File Content (Dual Pipeline)
```
Pipeline started at: 2025-10-02 19:19:47
SparkSession created successfully
Loading data from: ../../c4-train.00000-of-01024-30K.json
Document limit: 1000
Loaded 1000 records
Data loading time: 4417ms
Tokenizer configured
StopWordsRemover configured
HashingTF configured with 20000 features
IDF configured
Normalizer configured (L2 normalization)
Two pipelines configured: Basic (TF-IDF only) and Enhanced (TF-IDF + Normalizer)
Pipeline configuration time: 91ms
Training both pipelines for comparison...
Both pipelines trained successfully
Basic pipeline time: 3156ms
Enhanced pipeline time: 1307ms
Total pipeline training time: 4463ms
Reference Document (index 0): Beginners BBQ Class Taking Place in Missoula!
Do you want to get better at making delicious BBQ? You...

=== Document Similarity Analysis ===
Reference Document Index: 0
Reference Text: Beginners BBQ Class Taking Place in Missoula!
Do you want to get better at making delicious BBQ? You...

Top 5 Similar Documents:
1. Document 440 (Similarity: 0.08582317557998272)
   Text: Check out the Italian Pasta Salad calories and how many carbs in Italian Pasta Salad. Learn all the ...
2. Document 984 (Similarity: 0.07888578933197207)
   Text: Enjoy our variety of meat cuts prepared with authentic Southern Brazilian Style using our open fire ...
3. Document 54 (Similarity: 0.07839351684019706)
   Text: Know Buckeye Trail Class of 2001 graduates that are NOT on this List? Help us Update the 2001 Class ...
4. Document 278 (Similarity: 0.07389571072682868)
   Text: Class A Burn Prop - Stove Simulator With Overhead Burn Hood - Fire Facilities, Inc.
The stove simula...
5. Document 684 (Similarity: 0.07217586566622101)
   Text: Many students have never been to a museum before. While we want children to respect the museum and i...
Similarity analysis time: 1152ms
Saving pipeline comparison results to file...
Results saved successfully to results/lab17_pipeline_output.txt
File writing time: 1574ms
=== Dual Pipeline Performance Summary ===
Total documents processed: 1000
Document limit setting: 1000
Vocabulary size (HashingTF): 20000
Performance breakdown:
  Data loading: 4417ms
  Pipeline configuration: 91ms
  Basic pipeline training: 3156ms
  Enhanced pipeline training: 1307ms
  Similarity analysis: 1152ms
  File writing: 1574ms
  Total processing time: 11697ms
  Normalizer overhead: -1849ms
  Total pipeline time: 4463ms
Enhanced pipeline completed successfully at: 2025-10-02 19:20:01
Total execution time: 14108ms
Spark session closed successfully

```

## Enhanced Results Analysis

### Enhanced Statistics
- **Tổng số văn bản được xử lý**: Configurable via `limitDocuments`
- **Kích thước vector đặc trưng**: 20,000 chiều 
- **Pipeline comparison**: Basic TF-IDF vs Enhanced TF-IDF + Normalizer
- **Vector normalization**: L2 normalization cho enhanced pipeline
- **Cosine similarity**: Optimized với normalized vectors
- **Top similar documents**: 5 documents với highest cosine similarity

### Enhanced Output Format (Dual Pipeline Comparison)
```
Pipeline 1: Basic TF-IDF (no normalization)
Pipeline 2: Enhanced TF-IDF + L2 Normalization
==================================================

Document 1:
Text: Beginners BBQ Class Taking Place in Missoula!
Do you want to get better at making delicious BBQ? You...
Basic TF-IDF Features: (20000,[264,298,673,717,829,1271,1466,1499...],[....])
Normalized TF-IDF Features: (20000,[264,298,673,717,829,1271,1466...],[....])
------------------------------
Document 2:
Text: Discussion in 'Mac OS X Lion (10.7)' started by axboi87, Jan 20, 2012.
I've got a 500gb internal dri...
Basic TF-IDF Features: (20000,[406,643,651,1023,1349,1695...],[....])
Normalized TF-IDF Features: (20000,[406,643,651,1023,1349...],[....])
....


=== Document Similarity Analysis ===
Reference Document: Beginners BBQ Class Taking Place in Missoula!
Do you want to get better at making delicious BBQ? You...
Top 5 Similar Documents:
1. Document 440 (Similarity: 0.08582317557998272)
   Text: Check out the Italian Pasta Salad calories and how many carbs in Italian Pasta Salad. Learn all the ...
2. Document 984 (Similarity: 0.07888578933197207)
   Text: Enjoy our variety of meat cuts prepared with authentic Southern Brazilian Style using our open fire ...
3. Document 54 (Similarity: 0.07839351684019706)
   Text: Know Buckeye Trail Class of 2001 graduates that are NOT on this List? Help us Update the 2001 Class ...
4. Document 278 (Similarity: 0.07389571072682868)
   Text: Class A Burn Prop - Stove Simulator With Overhead Burn Hood - Fire Facilities, Inc.
The stove simula...
5. Document 684 (Similarity: 0.07217586566622101)
   Text: Many students have never been to a museum before. While we want children to respect the museum and i...
=== End of Output ===

```

### Kiểm Tra Kết Quả
```bash
# Kiểm tra file output đã được tạo
ls results/lab17_pipeline_output.txt
```

### Phân Tích Vector Đặc Trưng
- Biểu diễn thưa thớt giảm thiểu footprint bộ nhớ
- Giá trị khác không cho thấy mức độ quan trọng của từ sau khi áp dụng TF-IDF
- Giá trị cao hơn đại diện cho các từ đặc trưng hơn của mỗi văn bản

## Khó Khăn Gặp Phải và Giải Pháp
### 1. Tương Thích Phiên Bản Java
**Vấn đề**: 
- Vấn đề tương thích Java 19 với Spark 3.5.x
- Hạn chế truy cập module trong các phiên bản Java mới hơn

**Giải pháp**: 
- Thêm JVM arguments trong `build.sbt`:
```scala
javaOptions ++= Seq(
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  // ... thêm các opens khác
)
```

### 2. Giải Quyết Đường Dẫn
**Vấn đề**: 
- Vấn đề đường dẫn tương đối khi chạy từ các thư mục khác nhau
- Nhầm lẫn vị trí file dữ liệu

**Giải pháp**: 
- Sử dụng đường dẫn tương đối `../../c4-train.00000-of-01024-30K.json` từ thư mục
- Thêm logging phù hợp để theo dõi trạng thái tải file

### 3. Dual Pipeline Performance Optimization
**Vấn đề**: 
- Cần so sánh hiệu quả giữa basic TF-IDF và normalized TF-IDF
- Overhead của normalization step

**Giải pháp**: 
- Implement dual pipeline architecture để training parallel
- Đo performance metrics chi tiết cho từng pipeline
- Tính toán overhead percentage của normalization

### 4. Cosine Similarity Optimization  
**Vấn đề**:
- Cosine similarity calculation phức tạp cho raw TF-IDF vectors
- Performance issue với large vocabulary

**Giải pháp**:
- Sử dụng L2 normalized vectors: cosine similarity = dot product
- Caching intermediate results cho performance
- Optimized vector operations với Spark MLlib

## Kiến Trúc Code
### Cấu Trúc Class
```scala
object TextPipelineExample {
  def main(args: Array[String]): Unit = {
    // 1. Khởi tạo và thiết lập logging
    // 2. Tạo Spark session
    // 3. Tải và tiền xử lý dữ liệu
    // 4. Cấu hình các stage của pipeline
    // 5. Thực thi pipeline
    // 6. Lưu kết quả và logging
    // 7. Dọn dẹp và xử lý lỗi
  }
}
```

### Chiến Lược Xử Lý Lỗi
- Khối try-catch để quản lý ngoại lệ
- Graceful shutdown của Spark session
- Logging lỗi toàn diện
- Dọn dẹp tài nguyên trong khối finally

## Mở Rộng và Bài Tập

Codebase hỗ trợ bốn bài tập như được chỉ định:

1. **Chuyển Đổi Tokenizer**: Comment/uncomment các triển khai tokenizer thay thế
2. **Điều Chỉnh Feature Vector**: Sửa đổi tham số `numFeatures` (20000 → 1000)
3. **Mở Rộng Mô Hình ML**: Framework sẵn sàng để thêm LogisticRegression
4. **Tích Hợp Word2Vec**: Phương pháp vectorization thay thế đã được bao gồm

## Nhận Xét và Đánh Giá

### 1. Hiệu Quả Của Normalization
Việc bổ sung bước **L2 Normalization** trong Enhanced Pipeline đóng vai trò quan trọng trong việc cải thiện độ chính xác của thuật toán Cosine Similarity.
- **Trước khi chuẩn hóa**: Độ tương đồng bị ảnh hưởng lớn bởi độ dài văn bản (magnitude của vector).
- **Sau khi chuẩn hóa**: Độ tương đồng phản ánh chính xác hơn về mặt nội dung (góc giữa các vector), giúp tìm ra các văn bản có ngữ nghĩa gần nhau hơn thay vì chỉ là các văn bản có cùng độ dài hoặc chứa nhiều từ khóa lặp lại.

### 2. Đánh Giá Hiệu Năng (Performance)
Dựa trên log thực tế, việc thêm stage Normalizer có chi phí tính toán rất thấp so với lợi ích mang lại.
- Thời gian training của Enhanced Pipeline không chênh lệch đáng kể so với Basic Pipeline.
- Trong một số trường hợp chạy thực tế (như trong log mẫu), pipeline chạy sau có thể nhanh hơn do cơ chế caching và JVM warm-up của Spark, chứng tỏ kiến trúc này hoàn toàn khả thi cho dữ liệu lớn.

### 3. Khả năng Mở Rộng
Thiết kế **Dual Pipeline** cho phép dễ dàng A/B testing các kỹ thuật xử lý văn bản khác nhau (ví dụ: thay đổi Tokenizer, thay đổi số lượng features của HashingTF) mà không làm gián đoạn luồng xử lý chính.

## Kết Luận

Spark NLP pipeline đã triển khai thành công tất cả các thành phần yêu cầu:
- Nhập dữ liệu từ bộ dữ liệu C4
- Tiền xử lý văn bản với tokenization và loại bỏ từ dừng
- Vector hóa TF-IDF cho biểu diễn số
- L2 normalization cho vector đặc trưng
- Phân tích độ tương tự văn bản với cosine similarity
- Logging toàn diện và lưu trữ kết quả
- Xử lý lỗi và quản lý tài nguyên
