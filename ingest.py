# Simplified Concept for ingest.py
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load your 158 PDFs from a local folder
loader = PyPDFDirectoryLoader("./source_pdfs")
docs = loader.load()

# 2. Split them into chunks (e.g., paragraphs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 3. Create the Vector Database locally
# Note: You need a FREE Google API Key for embeddings
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="./chroma_db" # <--- This creates a folder of data
)

print("Ingestion complete! 'chroma_db' folder created.")