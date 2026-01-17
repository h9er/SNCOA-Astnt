import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings # <--- Needed to read the DB
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- COMPATIBILITY FIX ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 
# -------------------------

load_dotenv()

st.set_page_config(page_title="Instructor AI (Hybrid)", layout="wide")
CHROMA_PATH = "./chroma_db_hybrid"

@st.cache_resource
def load_vector_store():
    if not os.path.exists(CHROMA_PATH):
        return None
    
    # Key Change: Use Ollama to SEARCH the database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore

vectorstore = load_vector_store()

st.title("✈️ Air Force Instructor Assistant (Hybrid)")
st.caption("Engine: Google Gemini (Chat) + Ollama (Search) | 158 Documents Indexed")

if vectorstore is None:
    st.error("Database not found! Please run `python ingest_hybrid.py` first.")
else:
    # 1. Retriever (Uses Local CPU)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. LLM (Uses Google Cloud)
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    # Updated for 2026 Model Availability
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # 3. System Prompt
    system_prompt = (
        "You are an expert Air Force Instructor Assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, say you don't know. "
        "Always cite the document name if available."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    user_input = st.text_input("Ask a question about the source material:")

    if user_input:
        with st.spinner("Searching locally & reasoning in cloud..."):
            response = rag_chain.invoke({"input": user_input})
            st.markdown(f"### Answer:\n{response['answer']}")
            
            with st.expander("View Source Evidence"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.text(doc.page_content[:200] + "...")