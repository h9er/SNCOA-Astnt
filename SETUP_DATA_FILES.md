# 📦 Preparing Your Repository for Streamlit Cloud

The AI can't find sources because the data files aren't in your GitHub repository.

## Quick Fix

### Step 1: Make Sure Files Exist Locally
```bash
# Check if these directories exist:
ls source_docs/          # Should list PDF files
ls chroma_db_hybrid/     # Should list database files
```

### Step 2: Add Files to Git
```bash
# Add the data directories
git add source_docs/
git add chroma_db_hybrid/
git add .gitignore       # Ensures they stay in repo

# Verify they're staged
git status               # Should show them as "new file"

# Commit and push
git commit -m "Add course materials and vector database"
git push origin main
```

### Step 3: Verify on Streamlit Cloud
- The app will auto-redeploy
- Check the **System Status** panel in the sidebar
- Should show: ✅ source_docs and ✅ Vector DB
- System Status warning should disappear

---

## What's Happening

On your local machine:
- ✅ `source_docs/` has all the course PDFs
- ✅ `chroma_db_hybrid/` has the pre-built vector database
- ✅ The app works fine

On Streamlit Cloud (before fix):
- ❌ `source_docs/` is missing
- ❌ `chroma_db_hybrid/` is missing (default .gitignore was excluding them)
- ❌ No documents to retrieve → AI says "context not provided"

After adding to Git:
- ✅ Files are uploaded to Streamlit Cloud
- ✅ Vector store loads successfully
- ✅ AI can retrieve and answer questions

---

## Troubleshooting

**"Still no documents after pushing?"**
1. Double-check: `git status` shows the files committed
2. Wait 30 seconds and refresh the Streamlit app
3. Check app logs (Streamlit Cloud > Manage app > Logs)

**"Files are huge and slow?"**
- The PDFs (source_docs) can be large, but this is normal
- The vector database (chroma_db_hybrid) is usually 50-200MB
- Streamlit Cloud can handle this

**"Large file upload fails?"**
- GitHub has a 100MB per file limit
- If individual PDFs exceed 100MB, you need to split them
- The app has utilities to help with this

---

## Files That Need to Be in GitHub

```
SNCOA_Assistant/
├── main.py                    # ✅ Already there
├── requirements.txt           # ✅ Already there
├── source_docs/              # ❌ NEEDS TO BE ADDED (course materials)
├── chroma_db_hybrid/         # ❌ NEEDS TO BE ADDED (vector database)
├── .gitignore               # ✅ Just added (prevents accidental exclusion)
└── .streamlit/
    ├── config.toml          # ✅ Already there
    └── secrets.toml.example # ✅ Already there
```

---

Once these files are in your GitHub repo, Streamlit Cloud will have everything needed!
