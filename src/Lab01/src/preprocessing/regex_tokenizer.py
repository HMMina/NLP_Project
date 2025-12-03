import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
from typing import List
from core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens