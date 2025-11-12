import os
import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def retrieve_context(query: str, embedder, index, chunks: np.ndarray, top_k: int = 5) -> str:
    q_vec = embedder.encode([query])[0].astype("float32").reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)
    results = [chunks[i] for i in indices[0]]
    return "\n\n".join(results)

def generate_answer(prompt: str, llm_model: str) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful, precise support chatbot. If context is missing, ask a clarifying question before answering."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    cfg = load_config()
    print("🧠 Loading models & index...")
    embedder = SentenceTransformer(cfg["embedding_model"])
    index = faiss.read_index(cfg["index_file"])
    chunks = np.load("./data/chunks.npy", allow_pickle=True)
    llm_model = cfg["llm_model"]
    top_k = int(cfg["top_k"])

    print("🤖 RAG Chatbot ready. Type 'exit' to quit.")
    while True:
        try:
            user = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if user.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        context = retrieve_context(user, embedder, index, chunks, top_k=top_k)
        prompt = f"""Use the following context to answer accurately. If the context doesn't contain the answer, say so and ask a clarifying question.

Context:
{context}

Question: {user}
"""
        answer = generate_answer(prompt, llm_model)
        print(f"Bot: {answer}")

