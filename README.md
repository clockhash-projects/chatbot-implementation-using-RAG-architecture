# 🤖 RAG Chatbot (Python + FAISS)

A minimal, production-ready **Retrieval-Augmented Generation (RAG)** chatbot written in Python.
It embeds your FAQ/docs, stores vectors in **FAISS**, retrieves relevant chunks for each query, and
generates answers with an LLM (OpenAI-compatible by default).

> ✅ Great for prototypes, local demos, and as a base to evolve into ElasticSearch/Milvus/Pinecone or Agentic AI.

---

## 📁 Project Structure
rag_chatbot/

├── data/
│ └── docs/
│ ├── faq_getting_started.txt
│ ├── faq_account.txt
│ └── faq_billing.txt
├── build_index.py
├── chatbot.py
├── config.yaml
└── requirements.txt
