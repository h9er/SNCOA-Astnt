# Auto-Build Vector Store Feature

## Overview

The app now **automatically builds** the vector store (`chroma_db_hybrid`) from your course materials (`source_docs`) on first load. You no longer need to commit the pre-built database to GitHub!

## How It Works

### Storage Location

The vector store is built and stored at:
```
SNCOA_Assistant/
└── chroma_db_hybrid/          # Auto-created here on first load
    ├── chroma.sqlite3
    ├── [embedding data files]
```

**On Streamlit Cloud:** The vector store is created in the app's temporary filesystem during the first load.

**On Local Machine:** The vector store is persisted to disk in `chroma_db_hybrid/` for faster future loads.

### What You Need to Commit to GitHub

**Only this:**
```bash
git add source_docs/
git commit -m "Add course materials"
git push
```

**Do NOT commit:**
- ❌ `chroma_db_hybrid/` (auto-built, ~50-200MB)
- ❌ `chroma.sqlite3` (auto-generated)

### Build Process Timeline

On first Streamlit Cloud deploy:

1. **Startup** → App initializes
2. **Check** → "Does chroma_db_hybrid exist?" → No
3. **Load PDFs** → Reads all PDFs from `source_docs/` (~10-30 seconds)
4. **Split** → Breaks into chunks for better retrieval (~30 seconds)
5. **Embed** → Creates vector embeddings using Google AI (~1-2 minutes)
6. **Save** → Stores locally for future loads (~20 seconds)
7. **Ready** → Chat Assistant works!

**Total first-load time: ~2-3 minutes**
**Subsequent loads: ~5 seconds** (loads from cache)

## Configuration

The build process uses these settings (in `build_vector_store_from_pdfs()`):

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Size of each text chunk
    chunk_overlap=200,         # Overlap between chunks for context
    separators=["\n\n", "\n", " ", ""]
)
```

To adjust chunk size or overlap:
- **Larger chunks** (e.g., 2000) = fewer queries but more context per chunk
- **Smaller chunks** (e.g., 500) = more precise retrieval but more API calls
- **More overlap** = better semantic continuity, slower processing

## GitHub Deployment Steps

### Step 1: Clean Up Repository

```bash
# Remove the old vector store (it will auto-rebuild)
git rm -r chroma_db_hybrid/
git commit -m "Remove pre-built vector store (will auto-build)"

# Verify only source_docs/ is tracked
git status
```

Should show only `source_docs/` being tracked.

### Step 2: Ensure .gitignore Excludes Vector Store

Check that `.gitignore` contains:
```
chroma_db_hybrid/
chroma.sqlite3
```

### Step 3: Push to GitHub

```bash
git add source_docs/
git commit -m "Add course materials for auto-build"
git push origin main
```

### Step 4: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Deploy your app
3. On first load, watch the progress:
   - "🔨 Building vector store from PDFs..."
   - Progress bar shows: Loading → Splitting → Embedding → Saving
4. Chat Assistant becomes available after build completes

## Troubleshooting

### "Vector store building is too slow"
- **First load takes 2-3 minutes** (embedding all PDFs is slow)
- **Subsequent loads cache it** (should be instant)
- On Cloud, it rebuilds on each deploy (this is normal for stateless apps)

### "Out of memory" on Streamlit Cloud
- Streamlit Cloud has ~1GB RAM
- If you have massive PDFs (>500MB total), this can fail
- **Solution:** Split large PDFs before uploading to source_docs/

### "API rate limit exceeded"
- Google's embedding API has rate limits
- **Solution:** Wait a few minutes before redeploying

### "Still using old pre-built database?"
- Make sure you deleted `chroma_db_hybrid/` from GitHub
- Run: `git rm -r chroma_db_hybrid/`
- Then redeploy

## Monitoring

Check the System Status panel in the app sidebar:

| Status | Meaning |
|--------|---------|
| ✅ source_docs & ✅ Vector DB | Ready to use |
| ✅ source_docs & ❌ Vector DB | Building now (watch the main chat area for progress) |
| ❌ source_docs & ❌ Vector DB | Critical error - commit source_docs to GitHub |

## Local Development

The auto-build works locally too:

1. Make sure `source_docs/` has PDFs
2. Run: `streamlit run main.py`
3. Wait for build (or it uses cached version if already built)
4. Chat Assistant works immediately after

The `chroma_db_hybrid/` folder is automatically created in your workspace.

## Advanced: Manual Rebuild

To force a rebuild locally:

```bash
# Delete the cached vector store
rm -rf chroma_db_hybrid/

# Run the app - it will rebuild
streamlit run main.py
```

## Performance Notes

- **Build time:** 2-3 minutes for ~300 pages of PDFs
- **Retrieval time:** ~1 second to find relevant documents
- **API cost:** ~1-2 cents per first deployment (Google embeddings API)
- **Subsequent deploys:** Minimal cost (just querying Chroma locally)
