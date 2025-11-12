import os
import yaml
import faiss
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_docs(path: str) -> List[str]:
    docs = []
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                docs.append(text)
    return docs

def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    # simple word-based chunking; production systems may use tokenizer-based chunking
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

if __name__ == "__main__":
    cfg = load_config()
    docs_path = cfg["docs_path"]
    chunk_size = int(cfg["chunk_size"])
    index_file = cfg["index_file"]

    print("📄 Loading documents from:", docs_path)
    raw_docs = load_docs(docs_path)

    chunks: List[str] = []
    for d in raw_docs:
        chunks.extend(chunk_text(d, chunk_size=chunk_size))

    if not chunks:
        raise SystemExit("No documents found. Add files to ./data/docs and rerun.")

    print(f"🧩 Total chunks: {len(chunks)}")
    model_name = cfg["embedding_model"]
    print(f"🔢 Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    print("🧮 Generating embeddings...")
    vectors = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    # Build FAISS index (L2). For cosine sim, normalize first or use IndexFlatIP.
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_file)

    np.save("./data/chunks.npy", np.array(chunks, dtype=object))

    print("✅ Index built.")
    print(f"   • Index file : {index_file}")
    print("   • Chunks     : ./data/chunks.npy")

