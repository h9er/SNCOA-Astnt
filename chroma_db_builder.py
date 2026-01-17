import os
import shutil
import time
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb

# Load environment variables
load_dotenv()

# Configuration
SOURCE_DIRECTORY = "source_docs"
CHROMA_PATH = "chroma_db_hybrid"

# --- CHANGE 1: Smaller Batch Size (10 is 1/2 size and safer) ---
BATCH_SIZE = 10 

def safe_add_documents(vectorstore, documents, retries=8):
    """Adds documents with retry logic for 429 (Quota) and 504 (Timeout)."""
    for attempt in range(retries):
        try:
            vectorstore.add_documents(documents=documents)
            return  # Success! Exit the function
        except Exception as e:
            error_msg = str(e)
            
            # --- CHANGE 2: Catch 504 and Deadline Exceeded ---
            if any(x in error_msg for x in ["429", "Quota exceeded", "504", "Deadline Exceeded", "503"]):
                
                # Calculate wait time (exponential backoff: 2s, 4s, 8s...)
                wait_time = (2 ** attempt) + random.uniform(1, 3)
                
                # Extract specific retry time if Google provides it
                if "retry in" in error_msg:
                    try:
                        import re
                        match = re.search(r"retry in (\d+)", error_msg)
                        if match:
                            wait_time = float(match.group(1)) + 2
                    except:
                        pass
                
                error_type = "Timeout" if "504" in error_msg else "Rate Limit"
                print(f"   ⚠️ {error_type} (Attempt {attempt+1}/{retries}). Cooling down for {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # If it's a real crash (like a syntax error), fail immediately
                raise e
    print("❌ Failed after max retries.")

def build_database():
    # 1. Clear old database
    if os.path.exists(CHROMA_PATH):
        print(f"🗑️ Removing old {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)
    
    # 2. Initialize Embeddings
    print("🔑 Initializing Embeddings...")
    if "GOOGLE_API_KEY" not in os.environ:
        print("❌ Error: GOOGLE_API_KEY not found.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 3. Load PDFs
    print("📚 Loading PDFs...")
    docs = []
    if not os.path.exists(SOURCE_DIRECTORY):
        print(f"❌ Error: {SOURCE_DIRECTORY} not found.")
        return

    files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.endswith('.pdf')]
    for f in files:
        path = os.path.join(SOURCE_DIRECTORY, f)
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        print(f"   - Loaded {f}")

    # 4. Split Text
    print(f"✂️ Splitting {len(docs)} pages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   - Created {len(chunks)} text chunks")

    # 5. Create Persistent Client
    print("💾 Creating Persistent Database...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Initialize the collection (Empty first)
    vectorstore = Chroma(
        client=client,
        collection_name="sncoa_instructor_collection",
        embedding_function=embeddings,
    )

    # 6. BATCH PROCESSING
    total_batches = len(chunks) // BATCH_SIZE + 1
    print(f"🚀 Processing {len(chunks)} chunks in {total_batches} batches (Size: {BATCH_SIZE})...")
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"   Processing Batch {batch_num}/{total_batches}...", end="\r")
        
        safe_add_documents(vectorstore, batch)
        
        # Tiny sleep to be polite
        time.sleep(1.0)

    print("\n✅ Success! Database build complete.")
    print(f"   - Saved to '{CHROMA_PATH}'")

if __name__ == "__main__":
    build_database()