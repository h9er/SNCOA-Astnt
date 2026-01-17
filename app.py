import os
import sys
import streamlit as st

# --- COMPATIBILITY FIX ---
# This block attempts to swap the database driver.
# If it fails (because you are on Windows and didn't install the binary),
# it gracefully ignores the error and continues.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 
# -------------------------

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
# ... rest of your imports
# FIX: Override the system sqlite3 with the binary version we installed
# This is necessary for Streamlit Cloud / Linux servers
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # If running locally on Windows/Mac, this usually isn't needed

import streamlit as st
import chromadb # Now safe to import
# ... rest of your imports
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

st.title("SNCOA AI Assistant")

# 1. Load the pre-made database (Super fast!)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 matches

# 2. Setup the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3. Chat Interface
user_query = st.chat_input("Ask about a rubric or course material...")

if user_query:
    # Build the chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.invoke(user_query)
    
    st.write(response["result"])