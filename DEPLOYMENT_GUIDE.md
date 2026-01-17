# 🚀 SNCOA Instructor Assistant - Deployment Guide

## Local Development Setup

### Prerequisites
- Python 3.10+
- Google Gemini API Key (get it free at https://aistudio.google.com/apikey)
- Git

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/SNCOA_Assistant.git
   cd SNCOA_Assistant
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   - Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your-actual-api-key-here
   ```
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your API key

5. **Run Locally**
   ```bash
   streamlit run main.py
   ```

---

## Streamlit Cloud Deployment

### ❌ What Doesn't Work on Cloud
- `.env` files (they're local only)
- Ollama (requires server running)
- Vector store embeddings from Ollama (needs local Ollama)

### ✅ What We're Using Instead
- Google Gemini API (cloud-based)
- Pre-built Chroma vector database (stored in repo)

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git push origin main
   ```

2. **Create Secrets on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository
   - Once deployed, click "Manage app" → "Settings" → "Secrets"
   - Paste this:
   ```toml
   google_api_key = "your-actual-api-key-here"
   ```

3. **Verify It Works**
   - The app should now respond to queries
   - Look for debug messages showing:
     - ✅ AI model initialized
     - ✅ Chain ready
     - 📤 Streaming response from AI...
     - ✅ Response received (X chunks)

### 🆘 Troubleshooting

**Problem: "GOOGLE_API_KEY not found"**
- Solution: Add the secret to Streamlit Cloud Secrets (not GitHub Secrets)

**Problem: No response from AI (spinning forever)**
- Check debug messages to see where it's stuck
- Common causes:
  - API key is invalid
  - Gemini API is disabled in Google Cloud
  - Network timeout on Streamlit Cloud

**Problem: "Vector store unavailable"**
- This is expected on Streamlit Cloud (Ollama doesn't run there)
- The app should still work with Google's API
- If you see this but no answer, it's likely an API key issue

### 📊 Debug Output

When you ask a question, you'll see:
```
🔄 Thinking...
📡 Initializing AI model...
✅ AI model initialized
⛓️ Building response chain...
✅ Chain ready
🔍 Consulting 80 documents from: Module 1, Module 2, Module 3...
📤 Streaming response from AI...
✅ Response received (47 chunks)
```

If something fails, the error will be shown clearly with a traceback you can expand.

---

## Environment Variable Reference

### Required Variables
- `GOOGLE_API_KEY` - Your Google Gemini API key

### How to Get Google API Key
1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy the key
4. Add it to `.env` (local) or Streamlit Secrets (cloud)

---

## File Structure

```
SNCOA_Assistant/
├── main.py                  # Main Streamlit app
├── requirements.txt         # Python dependencies
├── .streamlit/
│   ├── config.toml         # Streamlit configuration
│   ├── secrets.toml.example # Template for secrets
│   └── secrets.toml        # YOUR SECRET KEY (gitignored)
├── source_docs/            # Course materials (PDFs)
├── chroma_db_hybrid/       # Pre-built vector database
└── DEPLOYMENT_GUIDE.md     # This file
```

---

## Performance Notes

- **First Load**: 2-3 seconds (model initialization)
- **Chat Response**: 5-15 seconds (depends on API latency)
- **Retrieval**: ~1 second (searching 80 documents)
- **Streaming**: ~3-8 seconds (depends on response length)

---

## Support

If you're still having issues:
1. Check that your API key is correct and Gemini API is enabled
2. Look at the debug output - it shows exactly where it's stuck
3. Try a simpler question first to verify the API connection
4. Check Streamlit Cloud app logs (click on your app, then "Logs")
