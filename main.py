import streamlit as st
import os
import sys
import warnings
import re
import html
import time

# --- 1. COMPATIBILITY FIX (CRITICAL FOR CLOUD) ---
# This must be the very first thing that runs
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass
# -------------------------------------------------

# --- 2. IMPORTS & SETUP ---
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Handle the LangChain split issues
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 3. CONFIGURATION ---
os.environ["ANONYMIZED_TELEMETRY"] = "False" 
os.environ["CHROMA_TELEMETRY_IMPL"] = "api.telemetry.NoOpTelemetry"
warnings.filterwarnings("ignore")

load_dotenv()

# --- CRITICAL: Handle Streamlit Secrets for Cloud Deployment ---
# On Streamlit Cloud, use st.secrets instead of .env
if "GOOGLE_API_KEY" not in os.environ and "google_api_key" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["google_api_key"]

# Verify API key exists before proceeding
if "GOOGLE_API_KEY" not in os.environ:
    st.error("""
    ❌ **GOOGLE_API_KEY not found!**
    
    For Streamlit Cloud deployment:
    1. Go to your GitHub repo Settings → Secrets and variables → Codespaces
    2. Add secret: `google_api_key` with your API key value
    
    For local testing:
    1. Create a `.env` file with: `GOOGLE_API_KEY=your_key_here`
    """)
    st.stop()

st.set_page_config(page_title="SNCOA Instructor Assistant", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY = os.path.join(current_dir, "source_docs")
CHROMA_PATH = os.path.join(current_dir, "chroma_db_hybrid")

# --- CRITICAL: Verify data directories exist on Cloud ---
source_docs_exists = os.path.exists(SOURCE_DIRECTORY)
chroma_exists = os.path.exists(CHROMA_PATH)

if not source_docs_exists or not chroma_exists:
    st.warning(f"""
    ⚠️ **Data Files Not Found on This Deployment**
    
    - source_docs: {'✅ Found' if source_docs_exists else '❌ Missing'}
    - chroma_db_hybrid: {'✅ Found' if chroma_exists else '❌ Missing'}
    
    **Solution for Streamlit Cloud:**
    Make sure these folders are committed to GitHub:
    ```bash
    git add source_docs/
    git add chroma_db_hybrid/
    git commit -m "Add data files"
    git push
    ```
    Then redeploy your app on Streamlit Cloud.
    """)

# --- 4. CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage { border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; }
    
    /* TOOLTIP STYLES - Fixed positioning to prevent cutoff */
    .tooltip {
        position: relative;
        display: inline-block;
        color: #d90429;
        font-weight: bold;
        cursor: help;
        border-bottom: 1px dotted #d90429;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 350px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        font-size: 0.85rem;
        font-weight: normal;
        position: absolute;
        z-index: 9999;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        max-height: 300px;
        overflow-y: auto;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        white-space: normal;
        word-wrap: break-word;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .stTextArea textarea { font-family: 'Courier New', monospace; }
    div.stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 5. ENGINE & DATA LOADING ---
@st.cache_resource
def load_vector_store():
    """Load Chroma vector store. Assumes database exists in repo."""
    
    # 1. Check if the folder exists on the server
    if not os.path.exists(CHROMA_PATH):
        st.error(f"❌ Error: 'chroma_db_hybrid' folder not found. Please ensure it is pushed to GitHub.")
        return None

    try:
        # 2. Force Google Embeddings
        # We must use the exact same model that built the database
        if "GOOGLE_API_KEY" not in os.environ:
             st.error("❌ GOOGLE_API_KEY missing. Cannot load vector store.")
             return None
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # 3. Connect using the Explicit Persistent Client
        # This is more robust for Cloud environments
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # 4. Load the Specific Collection
        # We must ask for "sncoa_instructor_collection" because that is what the builder script used.
        vectorstore = Chroma(
            client=client,
            collection_name="sncoa_instructor_collection",
            embedding_function=embeddings,
        )
        
        # 5. Verify Data
        # We access the internal collection to count the items
        count = vectorstore._collection.count()
        
        if count == 0:
            st.warning("⚠️ Database loaded, but it appears empty. Did the builder script save correctly?")
            # We do NOT trigger a rebuild here automatically. We want to see this warning.
            return None
        
        st.success(f"✅ Loaded {count} documents from pre-built database.")
        return vectorstore

    except Exception as e:
        st.error(f"❌ Critical Error loading database: {str(e)}")
        # We do NOT trigger a rebuild here. We need to see the error.
        return None
    
    # Initialize the global variable
vectorstore = load_vector_store()

@st.cache_data(show_spinner=False)
def read_pdf_directly(file_input):
    """ Reads one or multiple PDF/Text files and returns combined text. """
    if isinstance(file_input, str):
        file_list = [file_input]
    else:
        file_list = file_input

    combined_text = ""
    errors = []
    
    for filename in file_list:
        full_path = os.path.join(SOURCE_DIRECTORY, filename)
        if not os.path.exists(full_path):
            errors.append(f"[ERROR: File not found at {full_path}]")
            continue  # Continue instead of returning early
        try:
            if filename.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as f:
                    combined_text += f"\n--- PART: {filename} ---\n" + f.read()
            else:
                loader = PyPDFLoader(full_path)
                pages = loader.load()
                text = "\n".join([p.page_content for p in pages])
                combined_text += f"\n--- PART: {filename} ---\n" + text
        except Exception as e:
            errors.append(f"[ERROR Reading {filename}: {str(e)}]")
            continue  # Continue with next file instead of returning
    
    # If we had errors, prepend them but still return any content we did load
    if errors:
        combined_text = "\n".join(errors) + "\n" + combined_text
            
    return combined_text if combined_text else "[ERROR: No files were successfully loaded]"

# --- 6. CONSTANTS & MAPS ---
ACRONYM_MAP = {
    "AAFES": "Army and Air Force Exchange Service",
    "ACC": "Air Combat Command",
    "AFDP": "Air Force Doctrine Publication",
    "AFPC": "Air Force Personnel Center",
    "AOR": "area of responsibility",
    "C2": "Command and Control",
    "CVF": "Competing Values Framework (CVF) Quadrants Collaborate Create Control Compete",
    "DRU": "direct reporting unit",
    "FOA": "field operating agency",
    "JIPC": "joint imagery production complex",
    "laser": "light amplification by stimulated emission of radiation",
    "MAJCOM": "major command",
    "NAF": "numbered air force",
    "NATO": "North Atlantic Treaty Organization",
    "NCO": "Non-Commissioned Officer",
    "OODA": "Observe Orient Decide Act",
    "OPSEC": "operations security",
    "PACAF": "Pacific Air Forces",
    "P2": "Progressive Professionalism",
    "radar": "radio detection and ranging",
    "scuba": "self-contained underwater breathing apparatus",
    "SNCOA": "Senior Non-Commissioned Officer Academy",
    "USAFA": "United States Air Force Academy",
    "ZIP code": "Zone Improvement Plan code"
}

MODULE_FILE_MAP = {
    "Pre-Req": ["Pre-Req_combined.pdf"],
    "Module 0": ["Module 0_Combined.pdf"],
    "Module 1": ["Module 1_combined.pdf"],
    "Module 2": ["Module 2_combined-pages-1-pages-1.pdf",
                 "Module 2_combined-pages-1-pages-2.pdf",
                 "Module 2_combined-pages-1-pages-3.pdf",
                 "Module 2_combined-pages-1-pages-4.pdf",
                 "Module 2_combined-pages-2.pdf",
                 "Module 2_combined-pages-3.pdf",
                 "Module 2_combined-pages-4.pdf",
                ],
    "Module 3": ["Module 3_combined.pdf"],
    "Module 4": ["Module4_Primer Strategic Focus Lab 01 Aug 2024.pdf"]
}
CLOUD_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemma-3-27b-it", 
    "gemini-pro"
]

# --- 7. LOGIC FUNCTIONS ---

def get_llm_instance(engine_choice, model_name=None):
    """Create LLM instance with proper error handling."""
    try:
        if "Gemini" in engine_choice:
            target_model = model_name if model_name else "gemini-flash-latest"
            
            # Verify API key exists
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            llm = ChatGoogleGenerativeAI(model=target_model, temperature=0)
            return llm
        else:
            return ChatOllama(model="llama3.1", temperature=0)
    except Exception as e:
        error_msg = f"Failed to initialize LLM: {str(e)}"
        st.error(f"❌ {error_msg}")
        raise Exception(error_msg)

def grade_with_direct_read(rubric_filename, student_essay, prompt_template, content_scope, engine_choice):
    debug_log = [] 
    full_context_text = ""
    loaded_modules = []

    # --- 1. LOAD RUBRIC ---
    try:
        rubric_text = read_pdf_directly(rubric_filename)
        if "[ERROR" in rubric_text:
            debug_log.append(f"❌ Rubric Error: {rubric_text}")
        else:
            full_context_text += f"\n--- SOURCE: {rubric_filename} (RUBRIC) ---\n{rubric_text}\n"
            debug_log.append(f"✅ Loaded Rubric: {rubric_filename}")
            loaded_modules.append(rubric_filename)
    except Exception as e:
        debug_log.append(f"❌ Error loading Rubric: {str(e)}")

    # --- 2. LOAD CONTENT (MODULES) ---
    if content_scope:
        debug_log.append(f"📋 Loading {len(content_scope)} selected modules...")
        for module in content_scope:
            try:
                target_file = MODULE_FILE_MAP.get(module, module)
                text = read_pdf_directly(target_file)
                
                if "[ERROR" in text:
                    debug_log.append(f"❌ {module} Error: {text}")
                else:
                    # Handle both single files and lists of files
                    if isinstance(target_file, list):
                        file_str = f"{module} ({len(target_file)} parts)"
                        full_context_text += f"\n--- SOURCE: {file_str} (CONTENT) ---\n{text}\n"
                    else:
                        full_context_text += f"\n--- SOURCE: {target_file} (CONTENT) ---\n{text}\n"
                    
                    debug_log.append(f"✅ Loaded {module}")
                    loaded_modules.append(module)
            except Exception as e:
                debug_log.append(f"❌ Error loading {module}: {str(e)}")
    else:
        debug_log.append(f"⚠️ No content modules selected")

    # --- 3. LOAD STYLE GUIDES ---
    try:
        style_guide = "AFSNCOA Style Guide August 2025.pdf"
        text = read_pdf_directly(style_guide)
        if "[ERROR" not in text:
            full_context_text += f"\n--- SOURCE: {style_guide} (STYLE) ---\n{text}\n"
            debug_log.append(f"✅ Loaded Style Guide")
    except Exception as e:
        debug_log.append(f"❌ Style Guide skipped: {str(e)}")

    tnq_file = "DAFH33-337 Tongue and Quill Dec 22.pdf"
    if "Gemini" in engine_choice:
        try:
            text = read_pdf_directly(tnq_file)
            if "[ERROR" not in text:
                full_context_text += f"\n--- SOURCE: {tnq_file} (STYLE) ---\n{text}\n"
                debug_log.append(f"✅ Loaded Tongue & Quill (Cloud)")
        except:
            pass

    # --- 4. SAFETY CHECK: ESTIMATE SIZE ---
    # 1 Token ~= 4 Characters. 
    estimated_tokens = len(full_context_text) / 4
    debug_log.append(f"📊 Estimated Payload: {int(estimated_tokens):,} tokens")
    
    if estimated_tokens > 900000:
        debug_log.append("⚠️ WARNING: Approaching 1M token limit. This may cause 429 errors.")

    # --- 5. RETRY LOOP WITH BACKOFF ---
    if "Gemini" in engine_choice:
        models_to_try = CLOUD_MODELS
    else:
        models_to_try = ["llama3.1"]

    last_error = None
    
    for i, model_name in enumerate(models_to_try):
        try:
            debug_log.append(f"🔄 Attempt {i+1}: Grading with {model_name}...")
            
            llm = get_llm_instance(engine_choice, model_name)
            chain = prompt_template | llm | StrOutputParser()
            
            # STREAMING REQUEST
            stream = chain.stream({"context": full_context_text, "input": student_essay})
            
            # If we get here, connection was successful
            debug_log.append(f"✅ Success! Connected to {model_name}")
            return stream, debug_log
            
        except Exception as e:
            error_msg = str(e).lower()
            debug_log.append(f"❌ {model_name} failed: {str(e)[:100]}...")
            last_error = e
            
            # INTELLIGENT BACKOFF FOR 429 (Too Many Requests)
            if "429" in error_msg or "resource exhausted" in error_msg:
                wait_time = 10  # Wait 10 seconds to let the token bucket drain
                debug_log.append(f"⏳ Hit Rate Limit (429). Cooling down for {wait_time}s...")
                time.sleep(wait_time)
            
            continue
            
    # If we get here, all models failed
    raise last_error if last_error else Exception("All models failed silently.")

# --- 8. UI INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "rubric_query" not in st.session_state: st.session_state.rubric_query = ""
if "last_evidence" not in st.session_state: st.session_state.last_evidence = []

# --- 9. SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    
    # Engine Switcher
    engine_options = ["☁️ Gemini (Google)"]
    SHOW_LOCAL = False # <--- SET TO FALSE FOR CLOUD
    if SHOW_LOCAL:
        engine_options.append("🏠 Local (Ollama)")
        
    engine_choice = st.radio("Select AI Engine:", engine_options, help="Cloud mode is active.")
    st.markdown("---")
    
    mode = st.radio("Select Mission:", ["💬 Chat Assistant", "📝 Essay Grader"])
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.rubric_query = ""
        st.session_state.last_evidence = []
        st.rerun()
    
    # SYSTEM HEALTH CHECK
    st.markdown("---")
    with st.expander("🏥 System Status"):
        st.caption("**Data Files:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("source_docs", "✅" if source_docs_exists else "❌", 
                     help="Course materials folder")
        with col2:
            st.metric("Vector DB", "✅" if vectorstore else "❌",
                     help="Chroma database (auto-rebuilds from source_docs if missing)")
        
        st.caption("**Components:**")
        st.write(f"API Key: {'✅ Set' if 'GOOGLE_API_KEY' in os.environ else '❌ Missing'}")
        st.write(f"Vector Store: {'✅ Loaded' if vectorstore else '❌ Not loaded'}")
        
        if not vectorstore and not source_docs_exists:
            st.error("""
            **❌ Critical: Both Missing**
            
            Both source_docs and vector store are missing. 
            Make sure to commit to GitHub:
            ```
            git add source_docs/
            git commit -m "Add course materials"
            git push
            ```
            """)
        elif not vectorstore and source_docs_exists:
            st.warning("""
            **⚠️ Vector Store Missing**
            
            The app found source_docs but could not load the Database.
            Please check that 'chroma_db_hybrid' is on GitHub.
            """)
        elif vectorstore:
            st.success("""
            **✅ System Ready**
            
            All systems operational. The Chat Assistant can retrieve documents.
            """)
        
    # X-RAY TOOL
    st.markdown("---")
    with st.expander("🔍 X-Ray: Inspect PDF Text"):
        xray_key = st.selectbox("Select Module:", list(MODULE_FILE_MAP.keys()))
        file_entry = MODULE_FILE_MAP[xray_key]
        
        # For modules with multiple files, combine them
        if isinstance(file_entry, list):
            target_filenames = file_entry
            if len(file_entry) > 1:
                st.info(f"📄 {xray_key} contains {len(file_entry)} parts - showing combined view with overall page numbers")
        else:
            target_filenames = [file_entry]
        
        xray_page = st.number_input("Page Number (0-index):", min_value=0, value=0)
        
        if st.button("Read Page Text"):
            try:
                # Load all files for this module
                all_pages = []
                file_mapping = []  # Track which page comes from which file
                
                for filename in target_filenames:
                    full_path = os.path.join(SOURCE_DIRECTORY, filename)
                    loader = PyPDFLoader(full_path)
                    pages = loader.load()
                    for orig_page_num, page in enumerate(pages):
                        all_pages.append(page)
                        file_mapping.append({"file": filename, "original_page": orig_page_num})
                
                if xray_page < len(all_pages):
                    page_info = file_mapping[xray_page]
                    st.info(f"📖 **From:** {page_info['file']} (Original Page {page_info['original_page']})")
                    st.text_area("Raw Text Output:", all_pages[xray_page].page_content, height=300)
                else:
                    st.error(f"Page {xray_page} does not exist. This module has {len(all_pages)} total pages.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# --- 10. MAIN LAYOUT ---
st.title("✈️ SNCOA Instructor Assistant")
col_main, col_source = st.columns([0.65, 0.35], gap="medium")

# --- EVIDENCE BOARD (RIGHT COLUMN) ---
with col_source:
    st.markdown("### 📖 Evidence Board")
    if mode == "💬 Chat Assistant":
        st.caption("Referenced sources from last query")
        if st.session_state.last_evidence:
            for cite in st.session_state.last_evidence:
                with st.expander(f"[{cite['number']}] {cite['source']} - Page {cite['page']}"):
                    st.write(f"**Source:** {cite['source']}")
                    st.write(f"**Page:** {cite['page']}")
                    st.caption("Hover over citations [#] in the response to see sources")
        else:
            st.info("Ask a question to see sources here")
    elif mode == "📝 Essay Grader":
        st.caption("Evidence display is minimized in Direct Read mode.")

# --- MAIN INTERFACE (LEFT COLUMN) ---
with col_main:
    
    # === A. CHAT ASSISTANT ===
    if mode == "💬 Chat Assistant":
        st.header("Chat Assistant")
        
        # Display History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Input Logic
        if prompt := st.chat_input("Ask a question about the material..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🔄 Thinking...")
                debug_container = st.container()
                
                try:
                    # Step 1: Initialize LLM
                    debug_container.info("📡 Initializing AI model...")
                    llm = get_llm_instance(engine_choice)
                    debug_container.success("✅ AI model initialized")
                    
                    # 1. Retrieval (Safely handle if vectorstore is missing on Cloud)
                    context_text = ""
                    citations = []
                    docs_retrieved = 0
                    retrieval_warning = ""
                    
                    if vectorstore:
                        try:
                            # Increased k to ensure comprehensive coverage across all modules
                            # and better semantic matching for specific topics
                            retriever = vectorstore.as_retriever(search_kwargs={"k": 80})
                            docs = retriever.invoke(prompt)
                            docs_retrieved = len(docs)
                            
                            if not docs:
                                # Fallback: try with even broader retrieval
                                retriever = vectorstore.as_retriever(search_kwargs={"k": 120})
                                docs = retriever.invoke(prompt)
                                docs_retrieved = len(docs)
                            
                            # Build citations with source and page info - consolidate Module 2
                            seen_modules = {}
                            for doc in docs:
                                source = doc.metadata.get("source", "Unknown").split("\\")[-1]
                                page = doc.metadata.get("page", 0)
                                
                                # Detect if this is a Module 2 file and consolidate under "Module 2"
                                if "Module 2" in source:
                                    key = "Module 2"
                                else:
                                    key = source
                                
                                if key not in seen_modules:
                                    seen_modules[key] = {
                                        "pages": [page],
                                        "actual_file": source
                                    }
                                else:
                                    if page not in seen_modules[key]["pages"]:
                                        seen_modules[key]["pages"].append(page)
                                
                                context_text += f"\n[Citation: {source}, Page {page}]\n{doc.page_content}\n"
                            
                            # Convert consolidated citations to list with numbering
                            for idx, (module_name, info) in enumerate(seen_modules.items(), 1):
                                pages = sorted(info["pages"])
                                if len(pages) > 1:
                                    page_str = f"Pages {pages[0]}-{pages[-1]}"
                                else:
                                    page_str = f"Page {pages[0]}"
                                
                                citations.append({
                                    "number": idx,
                                    "source": module_name,
                                    "page": page_str
                                })
                        except Exception as retrieval_error:
                            retrieval_warning = f"⚠️ Retrieval failed: {str(retrieval_error)}"
                            message_placeholder.warning(f"{retrieval_warning}\nPlease try again or check system status.")
                    else:
                        retrieval_warning = "⚠️ Vector store not available (Ollama may not be running on Cloud). Using limited context."
                        message_placeholder.warning(retrieval_warning)
                    
                    # 2. Prompt with anti-hallucination guardrails and citation instruction
                    system_template = """You are an SNCOA Instructor Assistant designed to help military personnel learn from course materials.

                            YOUR PRIMARY RULES:
                            1. **ALWAYS GROUND RESPONSES IN PROVIDED CONTEXT** - Every claim must trace back to the provided materials or logical inference from them.
                            2. **NO HALLUCINATIONS** - Do NOT invent facts, statistics, names, or examples not in the context.
                            3. **ACKNOWLEDGE LIMITATIONS** - If information is not in the materials, explicitly say: "This is not covered in the provided materials" or "I cannot find this information in the sources."
                            4. **MAKE CONNECTIONS** - You ARE encouraged to:
                            - Connect concepts across different sources
                            - Explain how ideas relate to each other
                            - Help students draw conclusions from the material
                            - Ask clarifying questions if needed

                            CITATION REQUIREMENTS:
                            - Use [1], [2], etc. when directly quoting or referencing specific sources
                            - Example: "According to the doctrine [1], leadership involves..." 
                            - If synthesizing from multiple sources, cite each: "Combined with the framework in [1] and [2], we can conclude..."

                            RESPONSE STRUCTURE FOR COMPLEX QUESTIONS:
                            1. State what you CAN answer from the sources
                            2. Show the logical chain: "Source [X] says A, and source [Y] says B, so together they suggest C"
                            3. If asked for something not in materials, clearly state that and suggest what IS available
                            4. Do NOT fill gaps with assumptions or general knowledge

                            TONE:
                            - Professional and educational
                            - Honest about knowledge boundaries
                            - Encouraging analytical thinking
                            - Supporting student learning, not replacing it

                            CONTEXT (PROVIDED MATERIALS):
                            {context}
                            """
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_template),
                        ("user", "{input}")
                    ])
                    
                    # 3. Chain
                    debug_container.info("⛓️ Building response chain...")
                    chain = prompt_template | llm | StrOutputParser()
                    debug_container.success("✅ Chain ready")
                    
                    # 4. Show sources being consulted (transparency)
                    if docs_retrieved > 0 and citations:
                        sources_consulted = ", ".join([cite["source"] for cite in citations])
                        message_placeholder.info(f"🔍 Consulting {docs_retrieved} documents from: {sources_consulted}")
                    elif retrieval_warning:
                        message_placeholder.warning(retrieval_warning)
                    
                    # 5. Invoke & Stream with better error handling
                    try:
                        message_placeholder.empty()  # Clear the "Thinking..." message
                        full_response = ""
                        
                        # Stream the response
                        debug_container.info("📤 Streaming response from AI...")
                        chunk_count = 0
                        for chunk in chain.stream({"context": context_text, "input": prompt}):
                            chunk_count += 1
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")  # Add cursor effect
                        
                        debug_container.success(f"✅ Response received ({chunk_count} chunks)")
                        
                        # Final display without cursor
                        message_placeholder.empty()
                        
                        # 6. Add citations with tooltips to response
                        if full_response and full_response.strip() and citations:
                            # Store citations in session state for Evidence Board
                            st.session_state.last_evidence = citations
                            
                            # Build response with inline citation tooltips
                            response_with_citations = full_response
                            for cite in citations:
                                citation_marker = f"[{cite['number']}]"
                                tooltip_html = f"""<span class="tooltip">{citation_marker}<span class="tooltiptext"><strong>{cite['source']}</strong><br>Page {cite['page']}</span></span>"""
                                response_with_citations = response_with_citations.replace(citation_marker, tooltip_html)
                            
                            # Display response with hoverable citations
                            message_placeholder.markdown(response_with_citations, unsafe_allow_html=True)
                            
                            # 7. Save
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        elif full_response and full_response.strip():
                            st.session_state.last_evidence = []
                            message_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            message_placeholder.error("❌ No response received from AI. This may be a timeout or API issue. Please try again.")
                            debug_container.error("Stream completed but no content was returned")
                    except Exception as stream_error:
                        message_placeholder.error(f"❌ Stream Error: {type(stream_error).__name__}")
                        debug_container.error(f"Stream failed: {str(stream_error)}")
                        import traceback
                        debug_container.code(traceback.format_exc())
                    
                except Exception as e:
                    message_placeholder.error(f"❌ Error: {type(e).__name__}")
                    debug_container.error(f"Chat Error: {str(e)}")
                    import traceback
                    with debug_container.expander("Full Traceback"):
                        st.code(traceback.format_exc())

    # === B. ESSAY GRADER ===
    elif mode == "📝 Essay Grader":
        st.markdown("### Student Assessment")
        
        # Helper to set rubric
        def set_rubric(name): st.session_state.rubric_query = name
        
        # Quick Select Buttons
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.button("📄 Mod 1", on_click=set_rubric, args=["Module 1 Personal Leadership Assessment Evaluation with Template 1 Jan 2025.pdf"])
        with c2: st.button("📋 Mod 2", on_click=set_rubric, args=["Module 2 Background Paper 01 May 2024.pdf"])
        with c3: st.button("🛡️ Mod 3", on_click=set_rubric, args=["Module 3 - 1 Strategic Environment Brief Instructions and Eval Instrument August  2025.pdf"])
        with c4: st.button("📑 Mod 4 - Ref Strat", on_click=set_rubric, args=["Module 4 - AFSNCOA Personal Refinement Strategy 01 May 2024.pdf"])
        with c5: st.button("📋 Mod 4 - PoA", on_click=set_rubric, args=["Module 4 - Plan of Action Briefing 01 May 2024.pdf"])
        
        assignment_filename = st.text_input("Selected Rubric:", value=st.session_state.rubric_query, disabled=True)
        content_scope = st.multiselect("Include Content:", options=list(MODULE_FILE_MAP.keys()))
        student_essay = st.text_area("Paste Student Submission:", height=300)
        
        if st.button("Evaluate Submission", type="primary"):
            if not assignment_filename or not student_essay:
                st.warning("Please select a Rubric and paste the Essay.")
            else:
                result_container = st.container()
                result_container.info("⏳ Analyzing... (This uses Google's Brain, please wait 10s)")
                
                # GRADING TEMPLATE
                system_prompt = (
                    "You are 'CheckSix', an unforgiving, automated SNCOA Evaluation Engine. "
                    "You do NOT have a personality. You do NOT give 'advice'. "
                    "You exist only to compare the STUDENT ESSAY against the provided RUBRIC and SOURCE MATERIAL.\n\n"

                    "*** PHASE 1: MECHANICS SWEEP (CRITICAL) ***\n"
                    "Scan the text for these specific errors. QUOTE THE EXACT TEXT found in the essay so the student can Find (Ctrl+F) it.\n"
                    "1. **Dollar Signs:** Money must be written as '$100' or '100 dollars'. '100$' is an error.\n"
                    "2. **Punctuation:** Look for missing Oxford commas, comma splices, or double spaces.\n"
                    "3. **Spelling:** List any misspelled words.\n"
                    "4. **Incorrect Citations:** Flag incorrect citations.\n"
                    "5. **Passive Voice:** Identify over use of passive voice.\n\n"

                    "*** PHASE 2: CLAIM VERIFICATION ***\n"
                    "If the student makes a claim about the reading material (e.g., 'AFDP-1 states...'), verify it against the Context.\n"
                    "- If the source supports it -> MARK VERIFIED.\n"
                    "- If the source contradicts it -> MARK UNSUPPORTED and quote the real source text.\n\n"

                    "*** PHASE 3: SCORING (STRICT RUBRIC ADHERENCE) ***\n"
                    "Use the *exact* grading criteria from the loaded Rubric file.\n"
                    "- For Module 1: 0 Errors = 5 pts. 1-2 Errors = 4 pts. 3 Errors = 3 pts. 4+ Errors = 0 pts.\n"
                    "- For Module 2: 0 Errors = 10 pts. <4 Errors = 8 pts. 5+ Errors = 6 pts. Distracting = 0 pts.\n"
                    "- Apply deductions immediately based on the Mechanics Sweep count.\n\n"

                    "*** PHASE 4: OUTPUT FORMAT (MANDATORY) ***\n"
                    "Output the result in this Markdown format:\n\n"

                    "# 📊 ASSESSMENT REPORT\n\n"

                    "## 1. MECHANICS LOG\n"
                    "| Error Type | Quoted Text (Context) | Fix Recommendation |\n"
                    "|---|---|---|\n"
                    "| Format | \"...cost of 100$.\" | Move $ to front: '$100' |\n"
                    "| Grammar | \"...leaders, and followers.\" | Remove Oxford comma if not needed |\n"
                    "| Spelling | \"...definitly...\" | definitely |\n\n"
                    "| Citation | \"...Author & Title...\" | 1. AFSNCOA. Critical Thinking Primer. (Maxwell AFB-Gunter Annex AL, AFSNCOA, 2025) |\n\n"
                    "**TOTAL ERRORS FOUND:** (Count)\n"

                    "**GRAMMAR SCORE DEDUCTION:** (Explain based on Rubric rules)\n\n"

                    "## 2. CONTENT VERIFICATION\n"
                    "*(Check if the student's claims match the source material)*\n"
                    "- **Claim:** \"(Quote Student Claim)\"\n"
                    "  - **Verdict:** ✅ Verified / ❌ Unsupported\n"
                    "  - **Source Evidence:** \"(Quote the actual text from the Module that proves/disproves this)\"\n\n"

                    "## 3. RUBRIC SCORING GRID\n"
                    "*(Go through every criteria in the RUBRIC file. Assign points strictly.)*\n"
                    "| Rubric Criteria | Student Performance | Points/Grade |\n"
                    "|---|---|---|\n"
                    "| (e.g. Grammar/Format) | (e.g. Found 3 errors) | (e.g. 3/5) |\n"
                    "| (e.g. Use of Logic) | (e.g. Argument flows well...) | (e.g. 15/20) |\n\n"

                    "## 4. FINAL SCORE\n"
                    "**CALCULATED SCORE:** (Sum of points) / (Total Possible)\n"
                    "**INSTRUCTOR NOTE:** (A brief, stern summary of why they passed or failed.)"
                    "\n\n"
                    "CONTEXT DATA:\n{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt), 
                    ("human", "{input}")
                ])
                
                try:
                    stream, debug_logs = grade_with_direct_read(
                        assignment_filename, 
                        student_essay, 
                        prompt_template,
                        content_scope,
                        engine_choice
                    )
                    
                    # Clear the "Analyzing" message and write stream
                    result_container.empty()
                    
                    # Stream and capture response
                    full_response = ""
                    result_placeholder = result_container.empty()
                    
                    for chunk in stream:
                        full_response += chunk
                        result_placeholder.markdown(full_response)
                    
                    # Verify we got content
                    if not full_response or not full_response.strip():
                        result_container.error("No response received from AI. Please try again.")
                    
                    with st.expander("See Debug Logs"):
                        for log in debug_logs: st.write(log)
                        
                except Exception as e:
                    st.error(f"Grading Crash: {e}")