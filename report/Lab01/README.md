# NLP_APP_Lab1
## Cấu trúc dự án
```
 Lab1/
    src/
       core/
          interfaces.py         # Định nghĩa các interface Tokenizer và Vectorizer
          dataset_loaders.py    # Loader cho dữ liệu UD_English-EWT
       preprocessing/
          simple_tokenizer.py   # Tokenizer đơn giản
          regex_tokenizer.py    # Tokenizer sử dụng regex
       representations/
           count_vectorizer.py   # Vector hóa văn bản
    test/
        lab1_test.py              # Test cho Lab 1
        lab2_test.py              # Test cho Lab 2
 UD_English-EWT/                   # Dữ liệu tiếng Anh
```

## Lab 1: Tokenization
Lab 1 tập trung vào việc xây dựng các phương pháp tokenizer khác nhau để phân tách văn bản thành các token riêng lẻ:

### 1. Xây dựng interface
- Interface `Tokenizer` trong `src/core/interfaces.py` định nghĩa phương thức `tokenize` để phân tách văn bản

### 2. Cài đặt các tokenizer
- **SimpleTokenizer** trong `src/preprocessing/simple_tokenizer.py` sử dụng phương pháp đơn giản dựa trên việc thêm khoảng trắng vào trước và sau kí tự đặc biệt, sau đó phân tách văn bản dựa trên khoảng trắng.
- **RegexTokenizer** trong `src/preprocessing/regex_tokenizer.py` sử dụng biểu thức chính quy để lọc ra các token dựa trên mẫu `\w+|[^\w\s]`, giúp nhận diện từ và ký tự đặc biệt.


## Lab 2: Vector hóa văn bản (Count Vectorizer)
Lab 2 tập trung vào việc vector hóa văn bản sử dụng kỹ thuật Count Vectorizer:
### 1. Xây dựng interface
- Interface `Vectorizer` trong `src/core/interfaces.py` định nghĩa các phương thức cần thiết

### 2. Cài đặt Count Vectorizer
- `CountVectorizer` trong `src/representations/count_vectorizer.py` khởi tạo từ điển từ tập hợp các token
- Chuyển đổi văn bản thành vector đặc trưng dựa trên tần suất xuất hiện của các token

## Cách chạy code

### Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Chạy các file test trong VSCode (chạy trực tiếp trong IDE)
- Mở file `lab1_test.py` trong thư mục `test/` và chạy để kiểm tra Tokenizers
- Mở file `lab2_test.py` trong thư mục `test/` và chạy để kiểm tra Count Vectorizer

## Log kết quả
### Lab 1: Kết quả Tokenization
#### Testing Task 2/Lab1:
```
sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
```

```
Original: Hello, world! This is a test.
SimpleTokenizer: ["hello", ",", "world", "!", "this", "is", "a", "test", "."]
RegexTokenizer: ["hello", ",", "world", "!", "this", "is", "a", "test", "."]

Original: NLP is fascinating... isn't it?
SimpleTokenizer: ["nlp", "is", "fascinating", ".", ".", ".", "isn", "'", "t", "it", "?"]    
RegexTokenizer: ["nlp", "is", "fascinating", ".", ".", ".", "isn", "'", "t", "it", "?"]     

Original: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer: ["let", "'", "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]
RegexTokenizer: ["let", "'", "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]
```

#### Testing Task 3/Lab1:

```
Testing Task 3:
--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...

SimpleTokenizer Output (first 20 tokens): 
["al", "-", "zaman", ":", "american", "forces", "killed", "shaikh", "abdullah", "al", 
 "-", "ani", ",", "the", "preacher", "at", "the", "mosque", "in", "the"]

RegexTokenizer Output (first 20 tokens): 
["al", "-", "zaman", ":", "american", "forces", "killed", "shaikh", "abdullah", "al", 
 "-", "ani", ",", "the", "preacher", "at", "the", "mosque", "in", "the"]
```

### Lab 2: Kết quả Count Vectorizer
#### Testing Corpus:
```
corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
```

```
Count Vectorizer Test Corpus:
Vocabulary: {".", "a", "ai", "i", "is", "love", "nlp", "of", "programming", "subfield"}
Document-Term Matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

#### Testing với dữ liệu UD_English-EWT (5 dòng đầu tiên):

```
Count Vectorizer New Corpus from UD_English-EWT:
Vocabulary: {'!': 0, ',': 1, '-': 2, '.': 3, '2': 4, '3': 5, ':': 6, '[': 7, ']': 8, 'a': 9, 'abdullah': 10, 'al': 11, 'american': 12, 'ani': 13, 'announced': 14, 'at': 15, 'authorities': 16, 'baghdad': 17, 'be': 18, 'being': 19, 'border': 20, 'busted': 21, 'by': 22, 'causing': 23, 'cells': 24, 'cleric': 25, 'come': 26, 'dpa': 27, 'for': 28, 'forces': 29, 'had': 30, 'in': 31, 'interior': 32, 'iraqi': 33, 'killed': 34, 'killing': 35, 'ministry': 36, 'moi': 37, 'mosque': 38, 'near': 39, 'of': 40, 'officials': 41, 'operating': 42, 'preacher': 43, 'qaim': 44, 'respected': 45, 'run': 46, 'shaikh': 47, 'syrian': 48, 'terrorist': 49, 'that': 50, 'the': 51, 'them': 52, 'they': 53, 'this': 54, 'to': 55, 'town': 56, 'trouble': 57, 'two': 58, 'up': 59, 'us': 60, 'were': 61, 'will': 62, 'years': 63, 'zaman': 64}
Document-Term Matrix:
[0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
```


## Giải thích kết quả thu được
### Lab 1: Tokenization
- **SimpleTokenizer và RegexTokenizer** đều hiện thực hóa việc phân tách văn bản thành các token riêng lẻ
- Kết quả cho thấy cả hai tokenizer đều phân tách chính xác văn bản dựa trên khoảng trắng và dấu câu 
- Cả hai đều cho ra kết quả giống nhau trong các ví dụ thử nghiệm bởi logic phân tách đều được thiết kế để xử lý các trường hợp cơ bản như dấu câu và chữ
- **SimpleTokenizer** sử dụng phương pháp đơn giản:
  - Chuyển toàn bộ văn bản về chữ thường
  - Thêm khoảng trắng trước và sau các ký tự đặc biệt
  - Tách chuỗi dựa trên khoảng trắng

- **RegexTokenizer** sử dụng biểu thức chính quy:
  - Chuyển toàn bộ văn bản về chữ thường
  - Mẫu `\w+|[^\w\s]` giúp tìm kiếm các từ hoặc ký tự đặc biệt thỏa mãn điều kiện

### Lab 2: Count Vectorizer
- **CountVectorizer** chuyển đổi văn bản thành vector đặc trưng dựa trên tần suất xuất hiện của các token
- **Vocabulary** được xây dựng từ tất cả các token duy nhất trong corpus, sắp xếp theo thứ tự từ điển
- **Document-Term Matrix** hiển thị:
  - Mỗi dòng tương ứng với một văn bản
  - Mỗi cột tương ứng với một token trong từ điển
  - Mỗi giá trị là số lần xuất hiện của token trong văn bản
- Kết quả cho thấy văn bản đã được chuyển đổi thành dạng số học, có thể sử dụng cho các thuật toán học máy
- Tuy nhiên có thể thấy số chiều của vector khá lớn do số lượng token trong từ điển nhiều

## Khó khăn gặp phải và cách giải quyết
### Vấn đề import module
**Khó khăn:** 
- Khi chạy file test từ terminal, gặp lỗi:
  ```
  ModuleNotFoundError: No module named 'src'
  ```
- Các module không thể tìm thấy do đường dẫn import không phù hợp với cấu trúc thư mục
**Giải pháp:**
- Sử dụng đoạn code sau ở đầu file cần thiết:
  ```python
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  ```
- Sửa lại các import để phù hợp với cấu trúc thư mục

