import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import string
from typing import List
from core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, f" {p} ")
        tokens = text.split()
        return tokens