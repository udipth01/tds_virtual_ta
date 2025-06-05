import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding metadata
with open("embedding_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# Load FAISS index
index = faiss.read_index("my_index.faiss")

# Load sentence transformer model for query embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(query, top_k=5):
    query_emb = model.encode(query, convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_emb = query_emb.astype("float32")

    D, I = index.search(np.array([query_emb]), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        window = embedding_data[idx]
        results.append({
            "score": float(score),
            "topic_id": window["topic_id"],
            "topic_title": window["topic_title"],
            "root_post_number": window["root_post_number"],
            "post_numbers": window["post_numbers"],
            "combined_text": window["combined_text"],
        })
    return results

if __name__ == "__main__":
    query = input("Enter your question: ").strip()
    results = retrieve(query, top_k=3)

    print("\nTop retrieved subthreads:")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Score: {res['score']:.4f}")
        print(f"Topic ID: {res['topic_id']}, Root Post #: {res['root_post_number']}")
        print(f"Topic Title: {res['topic_title']}")
        print(f"Posts in subthread: {res['post_numbers']}")
        print("Content snippet:")
        print(res["combined_text"][:700], "...\n")
