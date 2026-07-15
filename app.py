import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PATH = Path("chroma_db")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_context(vectorstore, query, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n---\n\n".join([
        f"Source: {doc.metadata.get('source', 'CFPB Manual')} | Page {doc.metadata.get('page', '?')}\n{doc.page_content}"
        for doc in docs
    ])
    return context, docs

def ask_claude(query, context):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    system_prompt = """You are a knowledgeable assistant specializing in consumer financial protection law and mortgage regulations. 
You answer questions based strictly on the provided CFPB Supervision and Examination Manual context.
Always cite the page numbers from the source material when possible.
If the answer is not in the provided context, say so clearly rather than making up information.
Be precise and professional — your answers may be used by compliance professionals."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Use the following excerpts from the CFPB Supervision and Examination Manual to answer the question.

CONTEXT:
{context}

QUESTION: {query}

Please provide a clear, accurate answer based on the context above, citing relevant page numbers where applicable."""
            }
        ],
        system=system_prompt
    )
    return message.content[0].text

# Streamlit UI
st.set_page_config(
    page_title="CFPB Legal Q&A Agent",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ CFPB Mortgage Compliance Q&A Agent")
st.caption("RAG pipeline powered by LangChain + ChromaDB + Claude | Source: CFPB Supervision & Examination Manual")

st.markdown("""
**Ask questions about:**
- Mortgage servicing requirements
- Fair lending examination procedures  
- HMDA data collection and reporting
- Ability-to-repay and qualified mortgage rules
- RESPA and TILA compliance
""")

vectorstore = load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about CFPB mortgage regulations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual and generating answer..."):
            context, source_docs = retrieve_context(vectorstore, prompt)
            response = ask_claude(prompt, context)
            st.markdown(response)
            
            with st.expander("📄 Source passages retrieved"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Passage {i+1}** — Page {doc.metadata.get('page', '?')}")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})