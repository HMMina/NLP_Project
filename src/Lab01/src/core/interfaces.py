import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

class Vectorizer(ABC):
    """Abstract base class for vectorizers."""

    @abstractmethod
    def fit(self, corpus: List[str]):
        """Learn the vocabulary from a list of documents (corpus)."""
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """Transform a list of documents into a list of count vectors."""
        pass

    @abstractmethod
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """Fit the vectorizer on the corpus and return the transformed document-term matrix."""
        pass
