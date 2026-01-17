import os
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Connect
CHROMA_PATH = "./chroma_db_hybrid"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Get All IDs
print("🔍 Reading Database Inventory...")
data = vectorstore.get() # Grabs metadata for everything
unique_files = set()

for m in data['metadatas']:
    if m and 'source' in m:
        # Extract just the filename from the full path
        filename = m['source'].split(os.sep)[-1]
        unique_files.add(filename)

print(f"\n✅ FOUND {len(unique_files)} FILES:")
print("="*40)
for f in sorted(list(unique_files)):
    print(f'"{f}"')  # Printing with quotes so you see hidden spaces!
print("="*40)