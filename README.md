# 🤖 RAG Chatbot (Python + FAISS)

A minimal, production-ready **Retrieval-Augmented Generation (RAG)** chatbot written in Python.
It embeds your FAQ/docs, stores vectors in **FAISS**, retrieves relevant chunks for each query, and
generates answers with an LLM (OpenAI-compatible by default).

> ✅ Great for prototypes, local demos, and as a base to evolve into ElasticSearch/Milvus/Pinecone or Agentic AI.

---

# Step 1 – Add Your Documents

Place your content as plain text in:
data/docs/
Sample files included:

faq_getting_started.txt

faq_account.txt

faq_billing.txt

You can:

Edit these files

Add new .txt or .md files

Every file in data/docs/ will be indexed.

---
