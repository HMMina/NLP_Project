import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.count_vectorizer import CountVectorizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

dtm = vectorizer.fit_transform(corpus)

print("Count Vectorizer Test Corpus:")
print("Vocabulary:", vectorizer.vocabulary_)
print("Document-Term Matrix:")
for row in dtm:
    print(row)

new_vectorizer = CountVectorizer(tokenizer)
dataset_path = "C:\\Users\\ADMIN\\.vscode\\NLP_APP\\UD_English-EWT\\en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)
new_corpus = raw_text.split('\n')[:5]  # Lấy 5 dòng đầu tiên để thử nghiệm
new_dtm = new_vectorizer.fit_transform(new_corpus)
print("\nCount Vectorizer New Corpus from UD_English-EWT:")
print("Vocabulary:", new_vectorizer.vocabulary_)
print("Document-Term Matrix:")
for row in new_dtm:
    print(row)


