import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Environment setup (optional) ===
os.environ["HF_HOME"] = "D:/huggingface_cache"  # cache dir

# === Helper functions ===
def clean_text(text):
    return " ".join(text.strip().split())

def normalize(v):
    return v / np.linalg.norm(v)

# === Load your data ===
filename = "discourse_posts.json"  # Change as needed
with open(filename, "r", encoding="utf-8") as f:
    posts_data = json.load(f)

# === Group posts by topic_id ===
topics = {}
for post in posts_data:
    topic_id = post["topic_id"]
    if topic_id not in topics:
        topics[topic_id] = {"topic_title": post.get("topic_title", ""), "posts": []}
    topics[topic_id]["posts"].append(post)

# Sort posts by post_number within each topic
for topic_id in topics:
    topics[topic_id]["posts"].sort(key=lambda p: p["post_number"])

print(f"Loaded {len(posts_data)} posts across {len(topics)} topics.")

# === Initialize embedding model ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Build reply map & extract subthreads ===
def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    collected = []
    def dfs(post_num):
        post = posts_by_number[post_num]
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

# === Prepare embeddings for subthreads ===
embedding_data = []
embeddings = []

print("Building subthread embeddings...")

for topic_id, topic_data in tqdm(topics.items()):
    posts = topic_data["posts"]
    topic_title = topic_data["topic_title"]

    reply_map, posts_by_number = build_reply_map(posts)
    root_posts = reply_map[None]  # Root posts have no parent

    for root_post in root_posts:
        root_num = root_post["post_number"]
        subthread_posts = extract_subthread(root_num, reply_map, posts_by_number)

        combined_text = f"Topic title: {topic_title}\n\n"
        combined_text += "\n\n---\n\n".join(clean_text(p["content"]) for p in subthread_posts)

        emb = model.encode(combined_text, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)

        embedding_data.append({
            "topic_id": topic_id,
            "topic_title": topic_title,
            "root_post_number": root_num,
            "post_numbers": [p["post_number"] for p in subthread_posts],
            "combined_text": combined_text,
        })
        embeddings.append(emb)

embeddings = np.vstack(embeddings).astype("float32")

# Build FAISS index (cosine similarity via inner product on normalized vectors)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"Indexed {len(embedding_data)} subthreads.")

faiss.write_index(index, "my_index.faiss")
with open("embedding_data.json", "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, ensure_ascii=False, indent=2)

# === Retrieval function ===
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

# === Load generator model ===
gen_model_name = "google/flan-t5-base"  # smaller model for generation

tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

def generate_answer(query, retrieved_texts, max_length=256):
    context = "\n\n".join(retrieved_texts)
    prompt = f"Answer the question based on the following forum discussion excerpts:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
    outputs = gen_model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# === Example Query and usage ===
query = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
results = retrieve(query, top_k=3)

print("\nTop retrieved subthreads:")
for i, res in enumerate(results, 1):
    print(f"\n[{i}] Score: {res['score']:.4f}")
    print(f"Topic ID: {res['topic_id']}, Root Post #: {res['root_post_number']}")
    print(f"Topic Title: {res['topic_title']}")
    print(f"Posts in subthread: {res['post_numbers']}")
    print("Content snippet:")
    print(res["combined_text"][:700], "...\n")

retrieved_texts = [res["combined_text"] for res in results]
answer = generate_answer(query, retrieved_texts)

print("\nGenerated Answer:\n", answer)
