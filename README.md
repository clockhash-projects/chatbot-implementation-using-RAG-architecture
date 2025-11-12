# 🤖 RAG Chatbot (Python + FAISS)

A minimal, production-ready **Retrieval-Augmented Generation (RAG)** chatbot written in Python.
It embeds your FAQ/docs, stores vectors in **FAISS**, retrieves relevant chunks for each query, and
generates answers with an LLM (OpenAI-compatible by default).

> ✅ Great for prototypes, local demos, and as a base to evolve into ElasticSearch/Milvus/Pinecone or Agentic AI.

---

## Step 1 – Add Your Documents

Place your content as plain text in:
data/docs/

Sample files included:
- faq_getting_started.txt
- faq_account.txt
- faq_billing.txt

---

## Step 2 — Build the Vector Index
Generate the vector database for retrieval:
python build_index.py

This will produce:
- data/vector.index
- data/chunks.npy

## Step 3 — Run the Chatbot
Start the chatbot:
python chatbot.py

Example dialogue:
User: How do I reset my password?
Bot: You can reset your password from the Account → Security → Change Password page.


To exit, type:
exit

---

## How the RAG Pipeline Works

1. Load documents  
2. Chunk content  
3. Generate embeddings  
4. Build FAISS index  
5. Embed user query  
6. Retrieve top-K chunks  
7. Inject retrieved context  
8. LLM generates the answer

---

## License

MIT — free for personal and commercial use
