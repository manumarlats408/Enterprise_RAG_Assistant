from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

DATA_PATH = Path("data/documents")


def load_documents():
    documents = []
    for pdf in DATA_PATH.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)

    print(f"Documentos originales: {len(docs)}")
    print(f"Chunks generados: {len(chunks)}")
    print("\nPrimer chunk:\n")
    print(chunks[0].page_content[:500])
