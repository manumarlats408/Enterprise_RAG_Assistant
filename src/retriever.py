from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)


def query_vector_store(question, k=3):
    vector_store = load_vector_store()
    results = vector_store.similarity_search(question, k=k)
    return results


if __name__ == "__main__":
    question = input("Ingrese su pregunta: ")
    docs = query_vector_store(question)

    print("\n--- Resultados más relevantes ---\n")

    for i, doc in enumerate(docs):
        print(f"Resultado {i+1}:\n")
        print(doc.page_content[:500])
        print("\n" + "-"*50 + "\n")
