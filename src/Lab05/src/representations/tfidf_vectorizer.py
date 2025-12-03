import sys
import os
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, workspace_root)

from typing import List, Dict
import math
from Lab01.src.core.interfaces import Vectorizer, Tokenizer


class TfidfVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}
        self._num_documents: int = 0
        
    def fit(self, corpus: List[str]):
        # Build vocabulary
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        self._num_documents = len(corpus)
        
        # Calculate document frequency for each term
        doc_frequency = {token: 0 for token in self.vocabulary_}
        
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_doc_tokens = set(tokens)
            for token in unique_doc_tokens:
                if token in doc_frequency:
                    doc_frequency[token] += 1
        
        # Calculate IDF: log(N / df)
        for token, df in doc_frequency.items():
            # Add 1 to avoid division by zero (smoothing)
            self.idf_[token] = math.log((self._num_documents + 1) / (df + 1)) + 1
        
        return self
    
    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents to TF-IDF vectors."""
        vectors = []
        
        for doc in documents:
            # Initialize zero vector
            vector = [0.0] * len(self.vocabulary_)
            tokens = self.tokenizer.tokenize(doc)
            
            # Calculate term frequency for this document
            term_count = {}
            for token in tokens:
                if token in self.vocabulary_:
                    term_count[token] = term_count.get(token, 0) + 1
            
            # Calculate TF-IDF
            total_terms = len(tokens)
            if total_terms > 0:
                for token, count in term_count.items():
                    if token in self.vocabulary_:
                        tf = count / total_terms
                        idf = self.idf_.get(token, 0)
                        tfidf = tf * idf
                        vector[self.vocabulary_[token]] = tfidf
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        """Fit and transform in one step."""
        self.fit(corpus)
        return self.transform(corpus)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names (vocabulary)."""
        return sorted(self.vocabulary_.keys(), key=lambda x: self.vocabulary_[x])
