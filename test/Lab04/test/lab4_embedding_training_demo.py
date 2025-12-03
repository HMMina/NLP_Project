import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Đường dẫn file dữ liệu
DATA_PATH = "C:\\Users\\ADMIN\\.vscode\\NLP_APP\\UD_English-EWT\\en_ewt-ud-train.txt"
RESULT_PATH = "C:\\Users\\ADMIN\\.vscode\\NLP_APP\\Lab04\\results\\word2vec_ewt.model"


def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # simple_preprocess tự động chuyển về lowercase, bỏ ký tự đặc biệt
                yield simple_preprocess(line)


def main():
    print("=== Step 1: Đọc dữ liệu và tạo corpus stream ===")
    corpus = list(read_corpus(DATA_PATH))
    print(f"Số câu đọc được: {len(corpus)}")

    print("\n=== Step 2: Huấn luyện mô hình Word2Vec ===")
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,   # số chiều embedding
        window=5,          # context window
        min_count=2,       # bỏ các từ xuất hiện <2 lần
        workers=4,         # số luồng CPU
        sg=1               # 0 = CBOW, 1 = Skip-Gram
    )

    print("Huấn luyện xong!")

    print("\n=== Step 3: Lưu mô hình vào results/ ===")
    model.save(RESULT_PATH)
    print(f"Đã lưu mô hình tại: {RESULT_PATH}")

    print("\n=== Step 4: Demo sử dụng mô hình ===")
    # Ví dụ 1: tìm từ tương tự
    if "computer" in model.wv.key_to_index:
        print("\nTừ tương tự 'computer':")
        for word, score in model.wv.most_similar("computer", topn=5):
            print(f"  {word}: {score:.4f}")
    else:
        print("Từ 'computer' không có trong từ điển!")

    # Ví dụ 2: thử phép analogy: king - man + woman ≈ ?
    if all(w in model.wv.key_to_index for w in ["king", "man", "woman"]):
        print("\nPhép tương tự (king - man + woman):")
        result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
        for word, score in result:
            print(f"  {word}: {score:.4f}")
    else:
        print("Không đủ từ để thực hiện phép tương tự!")


if __name__ == "__main__":
    main()
