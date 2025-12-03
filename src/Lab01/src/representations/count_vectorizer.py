import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict
from core.interfaces import Vectorizer, Tokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}

    def fit(self, corpus: List[str]):
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        return self

    def transform(self, documents: List[str]) -> List[List[int]]:
        vectors = []
        for doc in documents:
            vector = [0] * len(self.vocabulary_)
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1
            vectors.append(vector)
        return vectors

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        self.fit(corpus)
        return self.transform(corpus)
