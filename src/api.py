from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

app = FastAPI(title="Enterprise RAG Assistant")

class QueryRequest(BaseModel):
    question: str


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )


def build_prompt(context, question):
    return f"""
You are an AI assistant answering questions strictly based on the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""


@app.post("/query")
def query_rag(request: QueryRequest):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(request.question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = OllamaLLM(model="mistral")
    prompt = build_prompt(context, request.question)

    answer = llm.invoke(prompt)

    return {
        "question": request.question,
        "answer": answer
    }
