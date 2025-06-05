import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# GitHub raw URLs
FAISS_URL = "https://raw.githubusercontent.com/udipth01/tds_virtual_ta/main/my_index.faiss"
EMBEDDING_DATA_URL = "https://raw.githubusercontent.com/udipth01/tds_virtual_ta/main/embedding_data.json"

# Helper function to download only if not present
def download_if_not_exists(url, local_filename):
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename}...")
        response = requests.get(url)
        with open(local_filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {local_filename}.")

# Download files once
download_if_not_exists(FAISS_URL, "my_index.faiss")
download_if_not_exists(EMBEDDING_DATA_URL, "embedding_data.json")

# Load data
with open("embedding_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

index = faiss.read_index("my_index.faiss")
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
