import sys
import os
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, workspace_root)

import gensim.downloader as api
import numpy as np
from Lab01.src.preprocessing.regex_tokenizer import RegexTokenizer


class WordEmbedder:
    def __init__(self, model_name: str):
        try:
            print(f"Đang tải mô hình: {model_name} ...")
            self.model = api.load(model_name)
            print("Mô hình đã được tải thành công.")
        except Exception as e:
            raise ValueError(f"Lỗi khi tải mô hình '{model_name}': {e}")

    def get_vector(self, word: str):
        """Trả về vector của từ, hoặc None nếu là OOV."""
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            print(f"Từ '{word}' không có trong từ điển (OOV).")
            return None

    def get_similarity(self, word1: str, word2: str):
        """Tính cosine similarity giữa hai từ."""
        if word1 not in self.model.key_to_index:
            print(f"'{word1}' không có trong từ điển.")
            return None
        if word2 not in self.model.key_to_index:
            print(f"'{word2}' không có trong từ điển.")
            return None
        return self.model.similarity(word1, word2)

    def get_most_similar(self, word: str, top_n: int = 10):
        """Trả về top_n từ tương tự nhất."""
        if word not in self.model.key_to_index:
            print(f"Từ '{word}' không có trong từ điển (OOV).")
            return []
        return self.model.most_similar(word, topn=top_n)

    def embed_document(self, document: str):
        """
        Biểu diễn toàn bộ văn bản bằng cách trung bình các vector từ.
        - Tokenize văn bản bằng RegexTokenizer.
        - Bỏ qua từ OOV.
        - Nếu không có từ hợp lệ → trả về vector 0 có cùng kích thước.
        """
        tokenizer = RegexTokenizer()
        tokens = tokenizer.tokenize(document)

        vectors = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)

        if len(vectors) == 0:
            # Không có từ hợp lệ → trả về vector 0 cùng kích thước
            dim = self.model.vector_size
            return np.zeros(dim)

        # Trả về vector trung bình của tất cả từ trong văn bản
        return np.mean(vectors, axis=0)
