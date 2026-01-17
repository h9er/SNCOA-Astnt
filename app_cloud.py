import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. COMPATIBILITY FIX (Cloud vs Local) ---
# This allows the app to work on Streamlit Cloud (Linux) and your PC (Windows)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 
# ---------------------------------------------

load_dotenv()

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="Instructor AI", layout="wide")
CHROMA_PATH = "./chroma_db_cloud"

# --- 3. LOAD THE BRAIN ---
@st.cache_resource
def load_vector_store():
    if not os.path.exists(CHROMA_PATH):
        st.error(f"Error: Database not found at {CHROMA_PATH}. Did you run ingest_cloud.py?")
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # "k": 5 means it retrieves the top 5 most relevant rubric/doc pages
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore

vectorstore = load_vector_store()

# --- 4. THE INTERFACE ---
st.title("📘 Air Force Instructor Assistant (Cloud PoC)")
st.caption("Using Google Gemini 1.5 Flash | Sourced from uploaded PDFs")

if vectorstore:
    # Set up the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Set up the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Define the System Prompt
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

    # Create the Processing Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Chat Input
    user_input = st.text_input("Ask a question about the Rubrics or Course Material:")

    if user_input:
        with st.spinner("Analyzing course documents..."):
            response = rag_chain.invoke({"input": user_input})
            
            # Display Answer
            st.success("Analysis Complete")
            st.markdown(f"### Answer:\n{response['answer']}")

            # Display Sources (Evidence)
            with st.expander("🔍 View Source Evidence"):
                for i, doc in enumerate(response["context"]):
                    source_name = doc.metadata.get('source', 'Unknown Source')
                    page_num = doc.metadata.get('page', 'Unknown Page')
                    st.markdown(f"**Source {i+1}:** *{source_name} (Page {page_num})*")
                    st.text(doc.page_content[:300] + "...") # Preview first 300 chars