import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DOCS_PATH = Path("docs")
CHROMA_PATH = Path("chroma_db")

def load_documents():
    docs = []
    for pdf_file in DOCS_PATH.glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        docs.extend(pages)
        print(f"  → {len(pages)} pages loaded")
    print(f"\nTotal pages: {len(docs)}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    return chunks

def create_vectorstore(chunks):
    print("\nCreating embeddings — this takes a few minutes...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )
    print(f"Vectorstore saved to {CHROMA_PATH}")
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)
    print("\nIngestion complete — ready to query!")