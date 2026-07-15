"""
app.py — Streamlit Chat Interface for CFPB Legal Q&A Agent
===========================================================
This module implements the conversational frontend and RAG retrieval logic
for the CFPB Mortgage Compliance Q&A Agent.
 
Architecture:
  1. User submits a natural language query via Streamlit chat input
  2. Query is embedded using the same model used at ingestion time
  3. ChromaDB performs cosine similarity search to retrieve top-K chunks
  4. Retrieved chunks are injected into Claude's context as grounded evidence
  5. Claude generates a response strictly grounded in the retrieved passages
  6. Source chunks are surfaced to the user for verification and transparency
 
This grounded retrieval approach prevents hallucination on regulatory questions
where accuracy and traceability are critical.
 
Prerequisites:
  - Run ingest.py once to build the vector store before launching this app
  - Set ANTHROPIC_API_KEY in .env file
 
Author: Alison Hartshorn
"""
 
import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 
# Load API key from .env
load_dotenv()
 
# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_PATH = Path("chroma_db")  # Must match path used in ingest.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Must match model used at ingestion
TOP_K_CHUNKS = 5     # Number of chunks to retrieve per query
MAX_TOKENS = 1024    # Maximum tokens in Claude's response
CLAUDE_MODEL = "claude-sonnet-4-6"
 
# ── System Prompt ─────────────────────────────────────────────────────────────
# Scoped to source material to prevent hallucination on regulatory content.
# Claude is explicitly instructed to acknowledge gaps rather than fabricate answers.
SYSTEM_PROMPT = """You are a knowledgeable compliance assistant specializing in 
consumer financial protection law and mortgage regulations. You answer questions 
based strictly on the provided CFPB Supervision and Examination Manual excerpts.
 
Guidelines:
- Always cite the page numbers from the source material when available
- If the answer is not in the provided context, clearly state that rather than 
  extrapolating or speculating
- Be precise and professional — your answers may be used by compliance professionals
- When relevant, note if regulations may have been updated since the manual was published
- Structure complex answers with clear sections for readability"""
 
 
@st.cache_resource
def load_vectorstore() -> Chroma:
    """
    Load the persisted ChromaDB vector store.
 
    Decorated with @st.cache_resource so the vector store is loaded once
    and reused across all user sessions — avoids reloading the embedding
    model on every page interaction.
 
    Returns:
        Chroma: Loaded vector store ready for similarity search.
 
    Raises:
        FileNotFoundError: If ingest.py has not been run to create the store.
    """
    if not CHROMA_PATH.exists():
        st.error(
            "Vector store not found. Please run `python ingest.py` first "
            "to process the source documents."
        )
        st.stop()
 
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
 
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )
 
 
def retrieve_context(vectorstore: Chroma, query: str, k: int = TOP_K_CHUNKS) -> tuple:
    """
    Retrieve the most semantically relevant chunks for a user query.
 
    Performs cosine similarity search between the query embedding and all
    stored document embeddings. Returns the top-K most similar chunks
    along with their metadata (source file, page number).
 
    Args:
        vectorstore: Loaded ChromaDB instance from load_vectorstore()
        query: User's natural language question
        k: Number of chunks to retrieve (default: TOP_K_CHUNKS)
 
    Returns:
        tuple: (formatted_context_string, list_of_source_documents)
            - context_string: Chunks formatted for Claude's prompt
            - source_docs: Raw Document objects for UI display
    """
    source_docs = vectorstore.similarity_search(query, k=k)
 
    # Format retrieved chunks with source metadata for Claude's context
    context_parts = []
    for doc in source_docs:
        source = doc.metadata.get("source", "CFPB Manual")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source} | Page {page}]\n{doc.page_content}"
        )
 
    context_string = "\n\n---\n\n".join(context_parts)
    return context_string, source_docs
 
 
def query_claude(user_question: str, context: str) -> str:
    """
    Generate a grounded answer using Claude with retrieved context.
 
    Constructs a prompt that provides retrieved CFPB manual passages as
    evidence and instructs Claude to base its answer strictly on that
    evidence. This RAG pattern grounds the LLM's response in source
    material, reducing hallucination risk on regulatory content.
 
    Args:
        user_question: The user's original natural language question
        context: Formatted string of retrieved document chunks
 
    Returns:
        str: Claude's grounded answer with source citations
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
 
    prompt = f"""Use the following excerpts from the CFPB Supervision and 
Examination Manual to answer the question below. Base your answer strictly 
on the provided context and cite relevant page numbers where applicable.
 
RETRIEVED CONTEXT:
{context}
 
QUESTION: {user_question}
 
Provide a clear, accurate, and professionally worded answer based on the 
context above. If the context does not contain sufficient information to 
fully answer the question, explicitly state what is and is not covered."""
 
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
 
    return response.content[0].text
 
 
# ── Streamlit UI ──────────────────────────────────────────────────────────────
 
st.set_page_config(
    page_title="CFPB Compliance Q&A Agent",
    page_icon="⚖️",
    layout="wide"
)
 
st.title("⚖️ CFPB Mortgage Compliance Q&A Agent")
st.caption(
    "RAG pipeline · LangChain + ChromaDB + Claude API · "
    "Source: CFPB Supervision & Examination Manual"
)
 
# Suggested questions to guide users on coverage
st.markdown("""
**This agent can answer questions about:**
- HMDA data collection and reporting requirements
- Fair lending examination procedures
- Ability-to-repay and qualified mortgage rules
- Mortgage servicing standards
- RESPA and TILA compliance requirements
- CFPB examination and supervision processes
""")
 
st.divider()
 
# Load vector store once (cached across sessions)
vectorstore = load_vectorstore()
 
# Initialize conversation history in session state
# Session state persists across Streamlit reruns within a single browser session
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Render existing conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# Handle new user input
if user_input := st.chat_input("Ask a question about CFPB mortgage regulations..."):
 
    # Display and store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching manual and generating answer..."):
 
            # Step 1: Retrieve relevant chunks via similarity search
            context, source_docs = retrieve_context(vectorstore, user_input)
 
            # Step 2: Generate grounded answer via Claude API
            answer = query_claude(user_input, context)
 
            # Display the answer
            st.markdown(answer)
 
            # Display retrieved source passages for transparency
            # Allows users to verify answers against the original manual
            with st.expander("📄 View retrieved source passages"):
                for i, doc in enumerate(source_docs, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Passage {i} — Page {page}**")
                    st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                    if i < len(source_docs):
                        st.divider()
 
    # Store assistant response in conversation history
    st.session_state.messages.append({"role": "assistant", "content": answer})
