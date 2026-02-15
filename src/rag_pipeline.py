from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


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
You are an AI assistant answering questions based only on the context below.

Context:
{context}

Question:
{question}

Answer in a concise and clear way.
"""


def run_rag(question, k=3):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(question, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = OllamaLLM(model="mistral")
    prompt = build_prompt(context, question)

    response = llm.invoke(prompt)
    return response


if __name__ == "__main__":
    question = input("Ingrese su pregunta: ")
    answer = run_rag(question)

    print("\n--- Respuesta del asistente ---\n")
    print(answer)
