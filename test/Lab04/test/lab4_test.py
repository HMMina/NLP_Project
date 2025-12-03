import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder
import numpy as np

def main():
    model_name = "glove-wiki-gigaword-50"
    embedder = WordEmbedder(model_name)

    print("\n--- EVALUATION: WORD EMBEDDING EXPLORATION ---")

    # 2️Lấy vector cho từ 'king'
    king_vec = embedder.get_vector("king")
    print("\nVector của 'king':")
    print(king_vec)

    # 3️ Tính độ tương đồng giữa 'king' và 'queen', và 'king' và 'man'
    sim_king_queen = embedder.get_similarity("king", "queen")
    sim_king_man = embedder.get_similarity("king", "man")
    print(f"\nĐộ tương đồng giữa 'king' và 'queen': {sim_king_queen:.4f}")
    print(f"Độ tương đồng giữa 'king' và 'man': {sim_king_man:.4f}")

    # 4️ Lấy 10 từ tương tự nhất với 'computer'
    most_similar = embedder.get_most_similar("computer", top_n=10)
    print("\nTop 10 từ tương tự với 'computer':")
    for w, score in most_similar:
        print(f"  {w}: {score:.4f}")

    # 5️ Biểu diễn văn bản "The queen rules the country."
    sentence = "The queen rules the country."
    doc_vec = embedder.embed_document(sentence)
    print("\nVector biểu diễn câu 'The queen rules the country.':")
    print(doc_vec)

if __name__ == "__main__":
    main()
