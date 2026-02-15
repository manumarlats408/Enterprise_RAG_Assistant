# Enterprise RAG Assistant

End-to-end Retrieval-Augmented Generation (RAG) system for querying enterprise documents using local embeddings and a local LLM.

## 🚀 Features

- PDF ingestion
- Semantic chunking
- Local embeddings (HuggingFace)
- FAISS vector database
- Local LLM via Ollama (Mistral)
- FastAPI endpoint (`/query`)
- Fully offline and cost-free

## 🧠 Architecture

User Question → FAISS Retrieval → Context Assembly → Local LLM → JSON Response

## ⚙️ Run Locally

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api:app --reload
