import sys
import os
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, workspace_root)

from typing import List
from sklearn.model_selection import train_test_split
from Lab01.src.preprocessing.regex_tokenizer import RegexTokenizer
from Lab01.src.representations.count_vectorizer import CountVectorizer
from Lab05.src.representations.tfidf_vectorizer import TfidfVectorizer
from Lab05.src.models.text_classifier import TextClassifier


def load_dataset() -> tuple[List[str], List[int]]:
    """Load sentiment analysis dataset."""
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    return texts, labels


def print_results(vectorizer_name: str, metrics: dict, y_test: List[int], y_pred: List[int], X_test: List[str]):
    """Print classification results."""
    print(f"\n{'='*60}")
    print(f"Results using {vectorizer_name}")
    print('='*60)
    
    print("\nPredictions:")
    for text, true_label, pred_label in zip(X_test, y_test, y_pred):
        true_sentiment = "positive" if true_label == 1 else "negative"
        pred_sentiment = "positive" if pred_label == 1 else "negative"
        status = "✓" if true_label == pred_label else "✗"
        print(f"  [{status}] Expected: {true_sentiment:8} | Predicted: {pred_sentiment:8}")
        print(f"      Text: {text}")
    
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.capitalize():10}: {value:.4f}")


def main() -> None:
    print("="*60)
    print("Text Classification: CountVectorizer vs TfidfVectorizer")
    print("="*60)
    
    texts, labels = load_dataset()
    print(f"\nDataset: {len(texts)} samples ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    tokenizer = RegexTokenizer()
    
    # Test 1: CountVectorizer
    print("\n" + "="*60)
    print("Test 1: Using CountVectorizer")
    print("="*60)
    
    count_vectorizer = CountVectorizer(tokenizer)
    count_classifier = TextClassifier(count_vectorizer)
    count_classifier.fit(X_train, y_train)
    count_predictions = count_classifier.predict(X_test)
    count_metrics = count_classifier.evaluate(y_test, count_predictions)
    
    print_results("CountVectorizer", count_metrics, y_test, count_predictions, X_test)
    
    # Test 2: TfidfVectorizer
    print("\n" + "="*60)
    print("Test 2: Using TfidfVectorizer")
    print("="*60)
    
    tfidf_vectorizer = TfidfVectorizer(tokenizer)
    tfidf_classifier = TextClassifier(tfidf_vectorizer)
    tfidf_classifier.fit(X_train, y_train)
    tfidf_predictions = tfidf_classifier.predict(X_test)
    tfidf_metrics = tfidf_classifier.evaluate(y_test, tfidf_predictions)
    
    print_results("TfidfVectorizer", tfidf_metrics, y_test, tfidf_predictions, X_test)
    
    # Comparison
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"{'Metric':<12} {'CountVectorizer':<18} {'TfidfVectorizer':<18} {'Winner':<10}")
    print("-"*60)
    
    for metric in count_metrics.keys():
        count_val = count_metrics[metric]
        tfidf_val = tfidf_metrics[metric]
        winner = "TF-IDF" if tfidf_val > count_val else "Count" if count_val > tfidf_val else "Tie"
        print(f"{metric.capitalize():<12} {count_val:<18.4f} {tfidf_val:<18.4f} {winner:<10}")


if __name__ == "__main__":
    main()
