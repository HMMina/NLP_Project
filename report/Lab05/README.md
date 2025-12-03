# Lab 05


## Báo cáo và Phân tích 

### 📋 Mục lục
1. [Các bước triển khai](#1-các-bước-triển-khai)
2. [Hướng dẫn chạy code](#2-hướng-dẫn-chạy-code)
3. [Phân tích kết quả](#3-phân-tích-kết-quả)
4. [Thách thức và Giải pháp](#4-thách-thức-và-giải-pháp)



## 1. Các bước triển khai

### 1.1 Task 1: Triển khai TextClassifier

**Bước 1: Thiết kế kiến trúc**
- Tạo class `TextClassifier` kết hợp:
  - Một vectorizer (CountVectorizer hoặc TfidfVectorizer)
  - Mô hình LogisticRegression từ scikit-learn
- Mục đích: Cung cấp interface thống nhất cho phân loại văn bản

**Bước 2: Triển khai các phương thức chính**
```python
class TextClassifier:
    def __init__(self, vectorizer: Vectorizer):
        """Khởi tạo với vectorizer tuân theo interface Vectorizer"""
        
    def fit(self, texts: List[str], labels: List[int]) -> None:
        """Huấn luyện mô hình trên văn bản đã gán nhãn"""
        # 1. Vector hóa các văn bản training bằng fit_transform
        # 2. Huấn luyện LogisticRegression trên các vector
        
    def predict(self, texts: List[str]) -> List[int]:
        """Dự đoán nhãn cho văn bản mới"""
        # 1. Vector hóa văn bản bằng transform (không phải fit_transform!)
        # 2. Trả về dự đoán từ LogisticRegression
        
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> dict:
        """Tính toán các độ đo hiệu suất"""
        # Trả về: accuracy, precision, recall, f1-score
```

**Bước 3: Cấu hình LogisticRegression**
- Max iterations: 100
- Random state: 42 (đảm bảo tính tái tạo)

**Các quyết định thiết kế quan trọng:**
- Sử dụng composition thay vì inheritance (has-a vectorizer, không phải is-a vectorizer)
- Tuân theo API fit/predict của scikit-learn
- Tách biệt concerns: vector hóa vs phân loại
- Hoạt động với bất kỳ implementation Vectorizer nào (interface Lab01)

### 1.2 Task 2: Chạy thử nghiệm với tập dữ liệu nhỏ
```
texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
```

### 1.3 Task 3: Pipeline với PySpark

**Bước 1: Thiết lập môi trường Spark**
```python
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
```

**Bước 2: Chuẩn bị dữ liệu**
- Load sentiments.csv (5,792 văn bản)
- Chuyển đổi nhãn sentiment: -1 → 0, +1 → 1 (phân loại nhị phân)
- Công thức: `label = (sentiment + 1) / 2`

**Bước 3: Xây dựng ML Pipeline**

Tạo pipeline 5 giai đoạn:
```python
Pipeline(stages=[
    Tokenizer(inputCol="text", outputCol="words"),
    StopWordsRemover(inputCol="words", outputCol="filtered_words"),
    HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000),
    IDF(inputCol="raw_features", outputCol="features"),
    LogisticRegression(maxIter=100, regParam=0.001)
])
```
**Pipeline gồm các thành phần:**
1. **Tokenizer**: Tách văn bản thành từ (xử lý khoảng trắng, dấu câu)
2. **StopWordsRemover**: Loại bỏ "the", "is", "and", v.v. (giảm nhiễu)
3. **HashingTF**: Trích xuất đặc trưng nhanh bằng hashing trick (không cần lưu vocabulary)
4. **IDF**: Gán trọng số cho các từ theo tầm quan trọng
5. **LogisticRegression**: Mô hình phân loại nhanh

**Bước 4: Chia Train-Test**
- Training: 80% (4,633 mẫu)
- Testing: 20% (1,159 mẫu)
- Split theo tỷ lệ (giữ nguyên phân phối class)

**Bước 5: Đánh giá**
- Sử dụng MulticlassClassificationEvaluator
- Các độ đo: accuracy, precision, recall, F1

### 1.4 Task 4: Các thí nghiệm cải thiện các thành phần trong mô hình
**Thí nghiệm 1: Điều chỉnh số chiều đặc trưng**
**Mục tiêu**: Tìm `numFeatures` tối ưu cho HashingTF

**Phương pháp**:
```python
feature_sizes = [1000, 5000, 10000, 20000]
for numFeatures in feature_sizes:
    # Xây dựng lại pipeline với numFeatures mới
    # Huấn luyện và đánh giá
    # Ghi nhận accuracy và thời gian training
```

**Kiểm định giả thuyết lựa chọn size vector**: 
- Quá nhỏ → Va chạm hash, mất thông tin
- Quá lớn → Đặc trưng thưa thớt, training chậm hơn

**Thí nghiệm 2: Word2Vec Embeddings**
**Mục tiêu**: Thay thế TF-IDF bằng semantic embeddings đặc
**Triển khai**:
```python
word2vec = Word2Vec(
    vectorSize=10000,        # Embeddings 10000 chiều
    minCount=5,              # Bỏ qua các từ hiếm
    inputCol="filtered_words",
    outputCol="features"
)
```

**Thí nghiệm 3: So sánh các mô hình phân loại**
**Các mô hình được kiểm tra**:
1. **Logistic Regression**
   - Biên quyết định tuyến tính
   - Training và inference nhanh
   - Dễ giải thích
   - Cấu hình: `maxIter=100, regParam=0.001`

2. **Naive Bayes**
   - Mô hình xác suất
   - Giả định các đặc trưng độc lập
   - Training rất nhanh
   - Cấu hình: `smoothing=1.0`

3. **Gradient-Boosted Trees (GBT)**
   - Tập hợp các cây quyết định
   - Biên quyết định phi tuyến
   - Xử lý tương tác giữa các đặc trưng
   - Cấu hình: `maxIter=20, maxDepth=7`

**Tiêu chí đánh giá**:
- Accuracy và F1-score (hiệu suất)
- Thời gian training (hiệu quả)
- Sử dụng bộ nhớ (khả năng mở rộng)

---

## 2. Hướng dẫn chạy code
### 2.1 Yêu cầu tiên quyết
**Cài đặt các thư viện cần thiết:**
```bash
pip install scikit-learn numpy pandas
pip install pyspark
```

**Thiết lập Dataset:**
Đảm bảo `sentiments.csv` tồn tại tại `C:\Users\ADMIN\.vscode\NLP_APP\sentiments.csv` với định dạng:
```
text,sentiment
"This is great!",1
"This is bad.",-1
```

### 2.2 Chạy các thí nghiệm
**Di chuyển đến thư mục Test:**
```bash
cd C:\Users\ADMIN\.vscode\NLP_APP\Lab05\test
```

#### Test 1: Phân loại văn bản cơ bản (Task 1 & 2)
**File:** `lab5_test.py`
**Mục đích**: So sánh CountVectorizer vs TfidfVectorizer trên tập dữ liệu nhỏ

**Lệnh:**
```bash
python lab5_test.py
```

**Kết quả thực thi:**
```
Text Classification: CountVectorizer vs TfidfVectorizer
============================================================

Dataset: 6 samples (3 positive, 3 negative)
Train samples: 4 | Test samples: 2

============================================================
Test 1: Using CountVectorizer
============================================================

============================================================
Results using CountVectorizer
============================================================

Predictions:
  [✗] Expected: positive | Predicted: negative
      Text: This movie is fantastic and I love it!
  [✗] Expected: negative | Predicted: positive
      Text: Could not finish watching, so bad.

Metrics:
  Accuracy  : 0.0000
  Precision : 0.0000
  Recall    : 0.0000
  F1        : 0.0000

============================================================
Test 2: Using TfidfVectorizer
============================================================

============================================================
Results using TfidfVectorizer
============================================================

Predictions:
  [✗] Expected: positive | Predicted: negative
      Text: This movie is fantastic and I love it!
  [✓] Expected: negative | Predicted: negative
      Text: Could not finish watching, so bad.

============================================================
Comparison Summary
============================================================
Metric       CountVectorizer    TfidfVectorizer    Winner
------------------------------------------------------------
Accuracy     0.0000             0.5000             TF-IDF
Precision    0.0000             0.0000             Tie
Recall       0.0000             0.0000             Tie
F1           0.0000             0.0000             Tie
```

**Kết luận:**
- Accuracy của cả hai mô hình thấp do tập dữ liệu rất nhỏ
- TfidfVectorizer nên vượt trội hơn CountVectorizer
- Cần thử nghiệm trên tập dữ liệu lớn hơn để có kết quả ý nghĩa

#### Test 2: Chạy thí nghiệm Pipeline với PySpark (Task 3)
**File:** `lab5_spark_sentiment_analysis.py`
**Mục đích**: Khởi tạo Pipeline PySpark và đánh giá mô hình phân tích cảm xúc

**Lệnh:**
```bash
python lab5_spark_sentiment_analysis.py
```

**Kết quả thực thi:**
```
============================================================
PySpark Sentiment Analysis Pipeline
============================================================
WARNING: Using incubator modules: jdk.incubator.vector
25/10/29 15:43:53 WARN Shell: Did not find winutils.exe: java.io.FileNotFoundException: java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset. -see https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/10/29 15:43:54 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable

============================================================
Step 1: Loading Data
============================================================
Initial rows: 5792

Sample data:
+--------------------------------------------------+---------+
|                                              text|sentiment|
+--------------------------------------------------+---------+
|Kickers on my watchlist XIDE TIT SOQ PNK CPW BP...|        1|
|user: AAP MOVIE. 55% return for the FEA/GEED in...|        1|
|user I'd be afraid to short AMZN - they are loo...|        1|
|                                 MNTA Over 12.00  |        1|
|                                  OI  Over 21.37  |        1|
+--------------------------------------------------+---------+
only showing top 5 rows
Rows after cleaning: 5791
Dropped 1 rows with null values

Label distribution:
+-----+-----+
|label|count|
+-----+-----+
|  0.0| 2106|
|  1.0| 3685|
+-----+-----+


============================================================
Step 2: Building Pipeline
============================================================
Stage 1: Tokenizer (text → words)
Stage 2: StopWordsRemover (words → filtered_words)
Stage 3: HashingTF (filtered_words → raw_features, 10000 features)
Stage 4: IDF (raw_features → features)
Stage 5: LogisticRegression (maxIter=100, regParam=0.001)

 Pipeline created with 5 stages

============================================================
Step 3: Training Model
============================================================
Training samples: 4682
Training in progress...
25/10/29 15:44:02 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
Model training completed!

============================================================
Step 4: Evaluating Model
============================================================
Test samples: 1109

 Sample predictions:
+--------------------------------------------------+-----+----------+-----------------------------------------+
|                                              text|label|prediction|                              probability|
+--------------------------------------------------+-----+----------+-----------------------------------------+
|  ISG An update to our Feb 20th video review..i...|  0.0|       1.0| [0.25869560846247547,0.7413043915375246]|
|  The rodeo clown sent BK screaming into the SI...|  0.0|       0.0| [0.999998410875766,1.589124233980499E-6]|
| , ES,SPY, Ground Hog Week, distribution at hig...|  0.0|       1.0| [0.035278463202358945,0.964721536797641]|
|                          ES, S  PAT TWO, update  |  0.0|       0.0|  [0.9971335766063395,0.0028664233936605]|
| PCN doulble top at key fib retracement weekly....|  0.0|       1.0|  [0.3992288932697887,0.6007711067302113]|
| also not very healthy, fell back below DT line...|  0.0|       1.0| [0.11947408164202711,0.8805259183579729]|
| thinking out loud. 50 mva sub 200 mva- done. B...|  1.0|       0.0| [0.99872844966236,0.0012715503376400372]|
|"RT @WSJheard: Canâ€™t get your hands on a Nint...|  1.0|       1.0| [0.00861611933751715,0.9913838806624828]|
|#ContrAlert Don't Panic: Wall Street Is Going C...|  0.0|       1.0|[0.006779450855187919,0.9932205491448121]|
|#CoronavirusPandemic As bad as #China's economi...|  0.0|       0.0| [0.9059362679382089,0.09406373206179108]|
+--------------------------------------------------+-----+----------+-----------------------------------------+
only showing top 10 rows

============================================================
Evaluation Metrics
============================================================
Accuracy : 0.7295
Precision: 0.7243
Recall   : 0.7295
F1 Score : 0.7248

============================================================
Prediction Analysis
============================================================
Total predictions: 1109
Correct: 809 (72.95%)
Incorrect: 300 (27.05%)

 Misclassified examples:
+--------------------------------------------------------------------------------+-----+----------+
|                                                                            text|label|prediction|
+--------------------------------------------------------------------------------+-----+----------+
|  ISG An update to our Feb 20th video review..if it closes below 495 much low...|  0.0|       1.0|
|                            , ES,SPY, Ground Hog Week, distribution at highs..  |  0.0|       1.0|
|               PCN doulble top at key fib retracement weekly....time to exit ...|  0.0|       1.0|
| also not very healthy, fell back below DT line after breaking it, SI weak, M...|  0.0|       1.0|
| thinking out loud. 50 mva sub 200 mva- done. Bottoming tails at 61.60 provid...|  1.0|       0.0|
| thinking out loud. 50 mva sub 200 mva- done. Bottoming tails at 61.60 provid...|  1.0|       0.0|
+--------------------------------------------------------------------------------+-----+----------+
+--------------------------------------------------------------------------------+-----+----------+
only showing top 5 rows
only showing top 5 rows

============================================================
Pipeline execution completed successfully!
============================================================

 Spark session stopped.
SUCCESS: The process with PID 23516 (child process of PID 3068) has been terminated.
SUCCESS: The process with PID 3068 (child process of PID 10716) has been terminated.
SUCCESS: The process with PID 10716 (child process of PID 20392) has been terminated.
```
**Kết luận:**
- Mô hình đạt ~73% accuracy trên tập test 1,109 mẫu
- Precision và Recall cân bằng tốt (~72-73%)
- Mô hình cải thiện hơn rất nhiều so với tập dữ liệu nhỏ ban đầu

#### Test 3: Cải thiện các thành phần mô hình (Task 4)
**File:** `lab5_improvement_test.py`
**Mục đích**: Thực hiện các thí nghiệm nhằm cải thiện hiệu suất mô hình

**Lệnh:**
```bash
python lab5_improvement_test.py
```


**Kết quả thực thi:**
```
======================================================================
TASK 4: Model Improvement Experiments
======================================================================
WARNING: Using incubator modules: jdk.incubator.vector
25/10/29 16:39:48 WARN Shell: Did not find winutils.exe: java.io.FileNotFoundException: java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset. -see https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/10/29 16:39:49 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable

Loading data from: c:\Users\ADMIN\.vscode\NLP_APP\sentiments.csv
Total samples: 5791
Train: 4682, Test: 1109

======================================================================
EXPERIMENT 1: Feature Dimensionality Reduction
======================================================================
Testing different numFeatures values: 1000, 5000, 10000, 20000

--- Testing with numFeatures = 1000 ---
25/10/29 16:40:05 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
  Accuracy: 0.7196
  F1 Score: 0.7156
  Training Time: 8.59s

--- Testing with numFeatures = 5000 ---
  Accuracy: 0.7277
  F1 Score: 0.7235
  Training Time: 4.33s

--- Testing with numFeatures = 10000 ---
  Accuracy: 0.7295
  F1 Score: 0.7248
  Training Time: 2.46s

--- Testing with numFeatures = 20000 ---
  Accuracy: 0.7358
  F1 Score: 0.7286
  Training Time: 2.53s

======================================================================
Experiment 1 Summary
======================================================================
NumFeatures     Accuracy     F1 Score     Train Time
----------------------------------------------------------------------
1000            0.7196       0.7156       8.59        s
5000            0.7277       0.7235       4.33        s
10000           0.7295       0.7248       2.46        s
20000           0.7358       0.7286       2.53        s

 Best: numFeatures=20000 with Accuracy=0.7358

======================================================================
EXPERIMENT 2: Word2Vec Embeddings
======================================================================
Using Word2Vec to generate dense word embeddings

Training Word2Vec model...

======================================================================
Experiment 2 Results
======================================================================
Accuracy : 0.7529
Precision: 0.7519
Recall   : 0.7529
F1 Score : 0.7409
Training Time: 39.12s

======================================================================
EXPERIMENT 3: Model Architecture Comparison
======================================================================
Comparing: Logistic Regression, Naive Bayes, GBT Classifier

--- Model 1: Logistic Regression ---
  Accuracy: 0.7295, F1: 0.7248, Time: 1.37s

--- Model 2: Naive Bayes ---
  Accuracy: 0.6844, F1: 0.6842, Time: 0.61s

--- Model 3: Gradient-Boosted Trees ---
  Accuracy: 0.7340, F1: 0.7042, Time: 19.80s

======================================================================
Experiment 3 Summary
======================================================================
Model                     Accuracy     F1 Score     Train Time
----------------------------------------------------------------------
Logistic Regression       0.7295       0.7248       1.37        s
Naive Bayes               0.6844       0.6842       0.61        s
Gradient-Boosted Trees    0.7340       0.7042       19.80       s

 Best Model: Gradient-Boosted Trees with Accuracy=0.7340

======================================================================
FINAL SUMMARY - All Experiments
======================================================================

 Key Findings:
1. Best numFeatures: 20000
2. Word2Vec Accuracy: 0.7529
3. Best Model: Gradient-Boosted Trees
SUCCESS: The process with PID 17620 (child process of PID 14724) has been terminated.
SUCCESS: The process with PID 14724 (child process of PID 24900) has been terminated.
SUCCESS: The process with PID 24900 (child process of PID 7936) has been terminated.
```

**Kết luận:**
- Thí nghiệm 1: Tăng numFeatures lên 20,000 đạt ~73.58% accuracy
- Thí nghiệm 2: Sử dụng Word2Vec embeddings đạt ~75.29% accuracy, tốt hơn TF-IDF
- Thí nghiệm 3: Gradient-Boosted Trees đạt ~73.40% accuracy, nhưng thời gian training lâu hơn nhiều so với Logistic Regression

---

## 3. Phân tích kết quả
### 3.1 Test 1
**Cấu hình mô hình:**
- **Vectorizer**: TfidfVectorizer và CountVectorizer
- **Classifier**: Logistic Regression (maxIter=100, regParam=0.001)
- **Dataset**: 6 mẫu (3 positive, 3 negative)

**Phân tích:**
- TF-IDF tốt hơn CountVectorizer: Đạt 50% accuracy so với 0% của CountVectorizer
- Tuy nhiên, cả hai đều có hiệu suất thấp do tập dữ liệu quá nhỏ (6 mẫu)
- Cần nhiều dữ liệu hơn để mô hình học tốt hơn

### 3.2 Test 2
**Cấu hình Pipeline:**
- Tokenizer: Tách văn bản thành từ (xử lý khoảng trắng, dấu câu)
- StopWordsRemover: Loại bỏ "the", "is", "and", v.v. (giảm nhiễu)
- HashingTF: Trích xuất đặc trưng nhanh bằng hashing trick (numFeatures=10,000)
- IDF: Gán trọng số cho các từ theo tầm quan trọng
- LogisticRegression : Mô hình phân loại (maxIter=100, regParam=0.001)
**Phân tích:**
- Mô hình đạt ~73% accuracy trên tập test 1,109 mẫu 
- Precision và Recall cân bằng tốt (~72-73%)
- Mô hình cải thiện hơn rất nhiều so với tập dữ liệu nhỏ ban đầu
- Pipeline khá hiệu quả trong việc xử lý văn bản và phân loại
- Các thành phần như StopWordsRemover và IDF đóng vai trò quan trọng trong việc nâng cao hiệu suất
- Có thể cải thiện thêm bằng cách điều chỉnh các thành phần trong pipeline (ví dụ: thử nghiệm với các tham số khác nhau cho HashingTF hoặc LogisticRegression, hay thử nghiệm thay thế các thành phần trong pipeline bằng các mô hình khác)

### 3.3 Test 3
**Thí nghiệm 1: Điều chỉnh số chiều đặc trưng**
Cấu hình Pipeline giống Test 2, thay đổi numFeatures của HashingTF

**Phân tích:**
- Tăng numFeatures từ 1,000 lên 20,000 cải thiện accuracy từ ~71.96% lên ~73.58%
- Do tập dữ liệu lớn có nhiều từ khác nhau, số chiều đặc trưng cao giúp giảm va chạm hash và giữ lại nhiều thông tin hơn
- Tuy nhiên, lợi ích giảm dần: Tăng từ 10,000 lên 20,000 chỉ cải thiện accuracy nhẹ (~0.63%)
- Cần cân nhắc giữa hiệu suất và chi phí tính toán để lựa chọn numFeatures phù hợp với dữ liệu

**Thí nghiệm 2: Word2Vec Embeddings**

Thay thế HashingTF + IDF bằng Word2Vec để tạo embeddings đặc

**Phân tích:**
- Mô hình đạt ~75.29% accuracy, vượt trội so với TF-IDF (~73.58%)
- Word2Vec mặc dù chỉ sử dụng vectorSize=10,000 nhưng đạt hiệu suất tốt tương đương với TF-IDF với numFeatures=20,000
- Điều này cho thấy embeddings đặc có thể nắm bắt ngữ nghĩa tốt hơn
- Tuy nhiên, thời gian training lâu hơn đáng kể (~39.12s so với ~2.53s của TF-IDF với numFeatures=20,000)
- Cần cân nhắc giữa hiệu suất và thời gian huấn luyện khi lựa chọn phương pháp trích xuất đặc trưng

**Thí nghiệm 3: So sánh các mô hình phân loại**

So sánh Logistic Regression, Naive Bayes và Gradient-Boosted Trees (GBT)

**Phân tích:**
- Logistic Regression đạt ~72.95% accuracy với thời gian training nhanh (~1.37s)
- Naive Bayes đạt ~68.44% accuracy, thấp hơn đáng kể so với hai mô hình còn lại, nhưng thời gian training rất nhanh (~0.61s)
- GBT đạt ~73.40% accuracy, tương đương với Logistic Regression, nhưng thời gian training lâu hơn nhiều (~19.80s)
-  Logistic Regression cân bằng tốt giữa hiệu suất và thời gian huấn luyện, là lựa chọn phù hợp cho bộ dữ liệu này, trong khi GBT có thể phù hợp hơn nếu ưu tiên hiệu suất hơn thời gian
- Naive Bayes có thể phù hợp hơn với các loại dữ liệu hoặc bài toán khác

### 3.4 Kết luận tổng thể
- TF-IDF với numFeatures cao (20,000) và Word2Vec embeddings đều mang lại hiệu suất tốt đối với bộ dữ liệu có kích thước lớn
- Logistic Regression là mô hình phân loại phù hợp cho bộ dữ liệu này, cân bằng tốt giữa hiệu suất và thời gian huấn luyện, trong khi GBT có thể được xem xét nếu ưu tiên hiệu suất hơn thời gian
- Việc điều chỉnh các thành phần trong pipeline (số chiều đặc trưng, phương pháp trích xuất đặc trưng, mô hình phân loại) đều ảnh hưởng đáng kể đến hiệu suất cuối cùng
- Cần cân nhắc kỹ lưỡng giữa hiệu suất, thời gian huấn luyện và chi phí tính toán khi thiết kế hệ thống

---

## 4. Thách thức và Giải pháp
### Thách thức 1: Lỗi import module
**Vấn đề:**
```python
ModuleNotFoundError: No module named 'Lab01'
```
**Nguyên nhân gốc:**
- Các script test Lab05 không tìm thấy modules Lab01
- Python path không bao gồm workspace root
**Điều tra:**
- Test scripts trong `Lab05/test/` cần đi lên 2 cấp, không phải 3
**Giải pháp:**
```python
# Trước (sai):
workspace_root = os.path.join(__file__, '..', '..', '..')

# Sau (đúng):
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, workspace_root)
```
**Bài học:**
- Luôn sử dụng `os.path.abspath()` cho đường dẫn tin cậy
- Test thiết lập `sys.path` bằng cách print `workspace_root`
- Cân nhắc sử dụng biến môi trường `PYTHONPATH` cho production

### Thách thức 2: Vấn đề import Lab01
**Vấn đề:**
```python
from src.core.interfaces import Vectorizer  # Thất bại
```
**Nguyên nhân gốc:**
- `count_vectorizer.py` của Lab01 sử dụng import tuyệt đối `from src.core.interfaces`
- Chỉ hoạt động nếu Lab01 trong `sys.path` như một package
- Gây ra vấn đề phụ thuộc vòng
**Giải pháp:**
```python
# Đã thay đổi trong Lab01/src/representations/count_vectorizer.py:
from core.interfaces import Vectorizer  # Import tương đối
```
**Bài học:**
- Ưu tiên relative imports trong một package
- Absolute imports chỉ dành cho dependencies bên ngoài

### Thách thức 3: Vấn đề bộ nhớ PySpark
**Vấn đề (Tiềm năng):**
```
Java.lang.OutOfMemoryError: Java heap space
```
**Nguyên nhân gốc:**
- Bộ nhớ driver mặc định của PySpark: 1GB
- Sentiment dataset + ML pipeline vượt mặc định
- Đặc biệt với numFeatures lớn (20K+)
**Giải pháp phòng ngừa:**
```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
```
**Bài học:**
- Luôn cấu hình bộ nhớ Spark cho ML workloads
- Bắt đầu với 4GB driver, tăng nếu cần
- Sử dụng Spark UI để debug vấn đề bộ nhớ

### Cải tiến tương lai

**Các nâng cấp tiềm năng:**
1. **Deep Learning**: Sử dụng LSTM hoặc BERT để hiểu ngữ cảnh tốt hơn
2. **Phương pháp Ensemble**: Kết hợp nhiều mô hình với voting
3. **Active Learning**: Gán nhãn lặp đi lặp lại các ví dụ không chắc chắn


---
## Tóm tắt
**Implementation hoàn thành:** 100%
- Task 1: TextClassifier với fit/predict/evaluate
- Task 2: TfidfVectorizer với công thức đúng
- Task 3: Pipeline phân tích cảm xúc PySpark
- Task 4: Ba thí nghiệm cải thiện mô hình

**Bài học chính:**
1. TF-IDF > CountVectorizer cho phân loại văn bản
2. Word2Vec nắm bắt ngữ nghĩa tốt hơn TF-IDF
3. Số chiều đặc trưng quan trọng nhưng có lợi ích giảm dần
4. Có thể lựa chọn nhiều loại mô hình phân loại tùy theo độ phù hợp với dữ liệu

