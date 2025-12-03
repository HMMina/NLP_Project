import sys
import os
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, workspace_root)

from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Lab01.src.core.interfaces import Vectorizer, Tokenizer
from Lab01.src.preprocessing.regex_tokenizer import RegexTokenizer
from Lab01.src.representations.count_vectorizer import CountVectorizer


class TextClassifier:
    def __init__(self, vectorizer: Vectorizer, *, solver: str = "liblinear", max_iter: int = 1000) -> None:
        self._vectorizer = vectorizer
        self._solver = solver
        self._max_iter = max_iter
        self._model: Optional[LogisticRegression] = None
        self._is_fitted: bool = False

    def fit(self, texts: List[str], labels: List[int]) -> "TextClassifier":
        features = self._vectorizer.fit_transform(texts)
        X = np.asarray(features)
        self._model = LogisticRegression(solver=self._solver, max_iter=self._max_iter, random_state=42)
        self._model.fit(X, labels)
        self._is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[int]:
        if not self._is_fitted or self._model is None:
            raise ValueError("The classifier must be fitted before making predictions.")

        features = self._vectorizer.transform(texts)
        X = np.asarray(features)
        predictions = self._model.predict(X)
        return predictions.tolist()

    @staticmethod
    def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
