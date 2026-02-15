from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from processing import load_documents, split_documents


def build_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local("vector_store")
    print("Vector store creado y guardado correctamente.")


if __name__ == "__main__":
    build_vector_store()
