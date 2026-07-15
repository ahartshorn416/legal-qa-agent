"""
ingest.py — Document Ingestion Pipeline for CFPB Legal Q&A Agent
=================================================================
This script handles the one-time setup of the RAG pipeline:
  1. Loads PDF documents from the /docs folder
  2. Splits them into overlapping chunks for retrieval
  3. Generates semantic embeddings using a sentence transformer model
  4. Persists the embeddings to a ChromaDB vector store
 
Run this script once before launching app.py, or re-run it
whenever source documents are added or updated.
 
Author: Alison Hartshorn
Source: CFPB Supervision and Examination Manual
        https://files.consumerfinance.gov/f/documents/cfpb_supervision-and-examination-manual.pdf
"""
 
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 
# Load environment variables from .env file
load_dotenv()
 
# ── Path Configuration ────────────────────────────────────────────────────────
DOCS_PATH = Path("docs")        # Directory containing source PDF documents
CHROMA_PATH = Path("chroma_db") # Persistent vector store location
 
# ── Chunking Configuration ────────────────────────────────────────────────────
# Chunk size of 1000 characters balances context richness with retrieval precision.
# Overlap of 200 characters ensures sentences spanning chunk boundaries
# are not lost — critical for regulatory text where context is everything.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
 
 
def load_documents() -> list:
    """
    Load all PDF documents from the docs/ directory.
 
    Uses LangChain's PyPDFLoader which extracts text page-by-page,
    preserving page number metadata for source citation in answers.
 
    Returns:
        list: LangChain Document objects, one per page across all PDFs.
    """
    documents = []
 
    pdf_files = list(DOCS_PATH.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCS_PATH}/. "
            "Download the CFPB Supervision and Examination Manual and place it there."
        )
 
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        documents.extend(pages)
        print(f"  → {len(pages)} pages loaded")
 
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents
 
 
def split_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for retrieval.
 
    RecursiveCharacterTextSplitter attempts to split on natural boundaries
    (paragraphs, then sentences, then words) before falling back to
    character-level splits — preserving semantic coherence in each chunk.
 
    Chunk overlap ensures that information near chunk boundaries is
    represented in both adjacent chunks, preventing retrieval gaps.
 
    Args:
        documents: List of LangChain Document objects from load_documents()
 
    Returns:
        list: Smaller Document chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split hierarchy: paragraphs → sentences → words → characters
        separators=["\n\n", "\n", ".", " "]
    )
 
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")
    return chunks
 
 
def create_vectorstore(chunks: list) -> Chroma:
    """
    Generate embeddings and persist them to a ChromaDB vector store.
 
    Uses the all-MiniLM-L6-v2 sentence transformer — a lightweight but
    highly effective model for semantic similarity tasks. It maps text
    to 384-dimensional vectors where semantic similarity corresponds
    to vector proximity (cosine similarity).
 
    ChromaDB persists embeddings to disk so the expensive embedding step
    only runs once. Subsequent app launches load from disk in seconds.
 
    Args:
        chunks: List of Document chunks from split_documents()
 
    Returns:
        Chroma: Populated vector store ready for similarity search.
    """
    print("\nInitializing embedding model (all-MiniLM-L6-v2)...")
    print("Note: First run downloads the model (~90MB). Subsequent runs use cache.")
 
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # Switch to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}  # Normalize for cosine similarity
    )
 
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    print("This may take several minutes on first run...")
 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )
 
    print(f"\nVector store persisted to: {CHROMA_PATH}/")
    print(f"Total vectors stored: {vectorstore._collection.count()}")
    return vectorstore
 
 
def main():
    """
    Orchestrates the full ingestion pipeline:
    Load → Split → Embed → Persist
    """
    print("=" * 60)
    print("CFPB Legal Q&A Agent — Document Ingestion Pipeline")
    print("=" * 60)
 
    # Step 1: Load source documents
    print("\n[1/3] Loading documents...")
    documents = load_documents()
 
    # Step 2: Split into retrieval-ready chunks
    print("\n[2/3] Splitting into chunks...")
    chunks = split_documents(documents)
 
    # Step 3: Embed and persist to vector store
    print("\n[3/3] Creating vector store...")
    create_vectorstore(chunks)
 
    print("\n" + "=" * 60)
    print("Ingestion complete. Run `streamlit run app.py` to start the agent.")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
