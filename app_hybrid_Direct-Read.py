import os
import sys
import warnings

# --- 0. SILENCE NOISE ---
os.environ["ANONYMIZED_TELEMETRY"] = "False" 
warnings.filterwarnings("ignore") 

import re
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# NEW: Import standard PDF loader
from langchain_community.document_loaders import PyPDFLoader

# --- 1. COMPATIBILITY FIX ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

load_dotenv()

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="SNCOA Instructor Assistant", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# !!! CRITICAL: UPDATE THIS PATH TO WHERE YOUR PDFs ACTUALLY LIVE !!!
# Use absolute path if possible, e.g., "C:/SNCOA_Assistant/source_files"
SOURCE_DIRECTORY = "C:/SNCOA_Assistant/source_docs"

CHROMA_PATH = "./chroma_db_hybrid"

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage { border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; }
    sup { color: #d90429; font-weight: bold; cursor: help; }
    [data-testid="column"]:nth-of-type(2) { background-color: rgba(248, 249, 250, 0.05); border-left: 1px solid #ddd; padding: 20px; height: 100vh; overflow-y: auto; }
    .stTextArea textarea { font-family: 'Courier New', monospace; }
    div.stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 4. ENGINE LOADING ---
@st.cache_resource
def load_vector_store():
    if not os.path.exists(CHROMA_PATH): return None
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

vectorstore = load_vector_store()

# --- 5. HELPER FUNCTIONS ---
def format_citations(text):
    return re.sub(r'\[(\d+)\]', r'<sup>\1</sup>', text)

# NEW: DIRECT DISK READER
def read_pdf_directly(filename):
    """
    Bypasses database. Opens PDF from disk and extracts ALL text.
    """
    full_path = os.path.join(SOURCE_DIRECTORY, filename)
    
    # Debug: Check if file exists
    if not os.path.exists(full_path):
        return f"[ERROR: File not found at {full_path}. Check SOURCE_DIRECTORY path.]"
    
    try:
        loader = PyPDFLoader(full_path)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
        return full_text
    except Exception as e:
        return f"[ERROR Reading PDF: {e}]"

# --- 6. SMART MODEL HANDLERS ---
MODEL_FALLBACK_LIST = ["gemini-2.0-flash-lite-preview-02-05", "gemini-flash-latest", "gemini-pro"]

MODULE_FILE_MAP = {
    "Pre-Req": "Pre-Req_combined.pdf",
    "Module 0": "Module 0_Combined.pdf",
    "Module 1": "Module 1_combined.pdf",
    "Module 2": "Module 2_combined.pdf",
    "Module 3": "Module 3_combined.pdf",
    "Module 4": "Module4_Primer Strategic Focus Lab 01 Aug 2024.pdf"
}

# HANDLER 1: FOR CHAT (Keeps using Database/Search)
def get_chat_response(input_text, retriever, prompt_template):
    status_placeholder = st.empty()
    for model_name in MODEL_FALLBACK_LIST:
        try:
            status_placeholder.info(f"🔄 Chatting with model: **{model_name}**...")
            fallback_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=0)
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(fallback_llm, prompt_template))
            response = chain.invoke({"input": input_text})
            status_placeholder.empty()
            return response, model_name
        except Exception:
            continue
    status_placeholder.error("❌ All models failed.")
    raise Exception("Model Error")

# HANDLER 2: FOR GRADING (Uses Direct Read)
def grade_with_direct_read(rubric_filename, student_essay, prompt_template, content_scope):
    status_placeholder = st.empty()
    debug_log = [] 
    full_context_text = ""

    # --- 1. READ RUBRIC (Direct) ---
    status_placeholder.info(f"📂 Opening Rubric: {rubric_filename}...")
    rubric_text = read_pdf_directly(rubric_filename)
    full_context_text += f"\n--- SOURCE: {rubric_filename} (RUBRIC) ---\n{rubric_text}\n"
    debug_log.append(f"✅ Loaded Rubric: {rubric_filename}")

    # --- 2. READ CONTENT (Direct) ---
    if content_scope:
        status_placeholder.info("📂 Opening Course Content...")
        for module in content_scope:
            target_file = MODULE_FILE_MAP.get(module, module)
            text = read_pdf_directly(target_file)
            full_context_text += f"\n--- SOURCE: {target_file} (CONTENT) ---\n{text}\n"
            debug_log.append(f"✅ Loaded Content: {target_file}")

    # --- 3. READ STYLE GUIDES (Direct) ---
    # Update these filenames to match yours exactly
    style_files = [
        "AFSNCOA Style Guide August 2025.pdf",
        "DAFH33-337 Tongue and Quill Dec 22.pdf"
    ]
    for sf in style_files:
        text = read_pdf_directly(sf)
        full_context_text += f"\n--- SOURCE: {sf} (STYLE) ---\n{text}\n"
        debug_log.append(f"✅ Loaded Guide: {sf}")

    # --- VISIBILITY ---
    with st.expander("🕵️ DEBUG: Files Loaded from Disk"):
        for entry in debug_log:
            st.write(entry)

    # --- 4. GENERATE GRADE ---
    for model_name in MODEL_FALLBACK_LIST:
        try:
            status_placeholder.info(f"📝 Grading with model: **{model_name}**...")
            fallback_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=0)
            chain = prompt_template | fallback_llm
            response = chain.invoke({"context": full_context_text, "input": student_essay})
            response_text = response.content if hasattr(response, 'content') else str(response)
            status_placeholder.empty()
            return response_text, model_name
        except Exception as e:
            time.sleep(0.5)
            continue
            
    status_placeholder.error("❌ All models failed.")
    raise Exception("Model Error")

# --- 7. UI SETUP ---
if "messages" not in st.session_state: st.session_state.messages = []
if "rubric_query" not in st.session_state: st.session_state.rubric_query = ""

with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Select Mission:", ["💬 Chat Assistant", "📝 Essay Grader"])
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.rubric_query = ""
        st.rerun()

st.title("✈️ SNCOA Instructor Assistant")
col_main, col_source = st.columns([0.65, 0.35], gap="medium")

# --- RIGHT COLUMN (Placeholder for Chat Mode) ---
with col_source:
    st.markdown("### 📖 Evidence Board")
    st.caption("In 'Grader Mode', the AI reads the full files directly from disk. Context is too large to display here.")

# --- LEFT COLUMN ---
with col_main:
    if mode == "💬 Chat Assistant":
        # (Standard Chat Logic preserved...)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(format_citations(message["content"]), unsafe_allow_html=True)
        if prompt := st.chat_input("Ask about rubrics..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            if vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                system_prompt = "You are an expert Assistant. Cite sources [1].\n\nCONTEXT:\n{context}"
                prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                try:
                    response, used_model = get_chat_response(prompt, retriever, prompt_template)
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

    elif mode == "📝 Essay Grader":
        st.markdown("### Student Assessment")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        def set_rubric(name): st.session_state.rubric_query = name
        
        # EXACT FILENAMES REQUIRED
        with col_r1: st.button("📄 Mod 1", on_click=set_rubric, args=["Module 1 Personal Leadership Assessment Evaluation with Template 1 Jan 2025.pdf"])
        with col_r2: st.button("📋 Mod 2", on_click=set_rubric, args=["Module 2 Background Paper 01 May 2024.pdf"])
        with col_r3: st.button("🛡️ Mod 3", on_click=set_rubric, args=["Module 3 - 1 Strategic Environment Brief Instructions and Eval Instrument August  2025.pdf"])
        with col_r4: st.button("🎯 Mod 4", on_click=set_rubric, args=["Module 4 - AFSNCOA Personal Refinement Strategy 01 May 2024.pdf"])

        assignment_filename = st.text_input("Selected Rubric File:", value=st.session_state.rubric_query, disabled=True)
        
        content_scope = st.multiselect("Include Content:", options=list(MODULE_FILE_MAP.keys()))
        student_essay = st.text_area("Paste Student Submission:", height=300)
        
        if st.button("Evaluate Submission", type="primary"):
            if not assignment_filename or not student_essay:
                st.warning("Please select a Rubric and paste the Essay.")
            else:
                system_prompt = (
                    "You are a strict Air Force Instructor. "
                    "Your job is to grade the student submission based on the FULL Rubrics, Content, and Style Guides provided."
                    "\n\n"
                    "INSTRUCTIONS:\n"
                    "0. **SOURCE ROLL CALL:** List the files you are reading.\n"
                    "1. **RUBRIC TABLE:** Create a table listing EVERY criteria in the rubric (Part I, II, III, etc) and the grade.\n"
                    "2. **NO ESTIMATION:** You have the full document. Do not estimate.\n"
                    "3. **CONTENT & REFS:** Verify facts and citations against the provided text.\n"
                    "\n\nCONTEXT (Full Files Loaded from Disk):\n{context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "STUDENT ESSAY:\n{input}")])

                try:
                    with st.spinner("Reading full files from disk..."):
                        response_text, used_model = grade_with_direct_read(
                            assignment_filename, 
                            student_essay, 
                            prompt_template,
                            content_scope
                        )
                        st.markdown("### Evaluation Result")
                        st.markdown(response_text)
                        st.caption(f"Graded by: {used_model}")
                except Exception as e:
                    st.error(f"Grading Failed: {e}")