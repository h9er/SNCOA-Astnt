import os
import sys
# Fix sqlite3 issue
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Connect to Database
CHROMA_PATH = "./chroma_db_hybrid"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# 2. Get all unique sources
print("🔍 Scanning Database Index...")
data = vectorstore.get()
sources = set()

for metadata in data['metadatas']:
    if metadata and 'source' in metadata:
        # Clean up the path to just show filename
        filename = metadata['source'].split(os.sep)[-1]
        sources.add(filename)

# 3. Print Report
print(f"✅ Found {len(sources)} unique documents.")
print("-" * 40)
found_target = False
target_snippet = "Module 1" # <--- Partial name to look for

for s in sorted(sources):
    print(f"📄 {s}")
    if target_snippet.lower() in s.lower():
        found_target = True

print("-" * 40)
if found_target:
    print(f"🎉 GREAT NEWS: found files matching '{target_snippet}'!")
else:
    print(f"❌ BAD NEWS: No file matching '{target_snippet}' was found.")
    print("   (It might have been skipped during ingest due to encryption or empty pages)")