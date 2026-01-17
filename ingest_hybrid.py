import os
# 1. SILENCE TELEMETRY (Must be before imports)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "./source_docs"
CHROMA_PATH = "./chroma_db_hybrid"

def generate_data_store():
    # --- STEP 1: LOAD ---
    print("✨ Loading PDFs using PyMuPDF...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    print(f"   - Loaded {len(documents)} pages.")

    # --- STEP 2: SPLIT ---
    print("✂️  Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    print(f"   - Created {total_chunks} text chunks.")

    # --- STEP 3: PREPARE DB ---
    print("💾 Initializing Local Database (Ollama)...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize the Vector Store with the Local Model
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_model
    )

    # --- STEP 4: BATCH PROCESS WITH PROGRESS BAR ---
    batch_size = 50 # Process 50 chunks at a time
    total_batches = (total_chunks // batch_size) + 1
    
    print(f"🚀 Starting Ingestion: ~{total_batches} batches to process.")
    print("   (This will take time. Please wait...)")
    print("-" * 50)

    start_time = time.time()

    for i in range(0, total_chunks, batch_size):
        # Create the batch
        batch = chunks[i : i + batch_size]
        
        # Add to DB
        vectorstore.add_documents(batch)
        
        # Calculate Progress
        current_batch = (i // batch_size) + 1
        percent = int((current_batch / total_batches) * 100)
        elapsed = int(time.time() - start_time)
        
        # Print Update
        print(f"   ✅ Batch {current_batch}/{total_batches} Complete ({percent}%) - Elapsed: {elapsed}s")

    print("-" * 50)
    print(f"🎉 Success! All {total_chunks} chunks saved to '{CHROMA_PATH}'.")

if __name__ == "__main__":
    generate_data_store()