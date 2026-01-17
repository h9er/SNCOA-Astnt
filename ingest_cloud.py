import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "./source_docs"
CHROMA_PATH = "./chroma_db_cloud"

def generate_data_store():
    print("✨ Loading PDFs using PyMuPDF...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    print(f"   - Loaded {len(documents)} pages.")

    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   - Created {len(chunks)} text chunks.")

    print("💾 Initializing ChromaDB...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # 1. Initialize the Database (but don't add data yet)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    # 2. Add chunks in small batches to respect Rate Limits
    batch_size = 50  # Send 50 chunks at a time
    total_batches = len(chunks) // batch_size + 1
    
    print(f"🚀 Starting Ingestion: Processing {len(chunks)} chunks in ~{total_batches} batches.")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        # Retry logic in case of a random hiccup
        try:
            vectorstore.add_documents(batch)
            print(f"   - Processed batch {i // batch_size + 1}/{total_batches} ({len(batch)} chunks)")
            
            # CRITICAL: Sleep to reset the API quota limit
            time.sleep(2.0) 
            
        except Exception as e:
            print(f"   ⚠️ Error on batch {i}: {e}")
            print("   Waiting 60 seconds to cool down...")
            time.sleep(60) # If we hit a limit, wait a full minute
            try:
                vectorstore.add_documents(batch)
                print("   - Retry successful!")
            except Exception as e2:
                print(f"   ❌ Failed batch permanently: {e2}")

    print(f"✅ Success! Database saved to '{CHROMA_PATH}'.")

if __name__ == "__main__":
    generate_data_store()