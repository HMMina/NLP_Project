import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer

sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

print("Testing Task 2:")
for sentence in sentences:
    print(f"Original: {sentence}")
    print("SimpleTokenizer:", simple_tokenizer.tokenize(sentence))
    print("RegexTokenizer:", regex_tokenizer.tokenize(sentence))
    print()

print("Testing Task 3:")
from src.core.dataset_loaders import load_raw_text_data
# ... (your tokenizer imports and instantiations) ...
dataset_path = "C:\\Users\\ADMIN\\.vscode\\NLP_APP\\UD_English-EWT\\en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)
# Take a small portion of the text for demonstration
sample_text = raw_text[:500] # First 500 characters
print("--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text[:100]}...")
simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
regex_tokens = regex_tokenizer.tokenize(sample_text)
print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")