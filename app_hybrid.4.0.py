import os
import sys
import warnings
import re
import time
import html
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# --- 0. SILENCE NOISE (UPDATED) ---
os.environ["ANONYMIZED_TELEMETRY"] = "False" 
os.environ["CHROMA_TELEMETRY_IMPL"] = "api.telemetry.NoOpTelemetry" # <--- KILLS THE CHROMA ERROR
warnings.filterwarnings("ignore")

# --- 1. COMPATIBILITY FIX ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

load_dotenv()

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="SNCOA Instructor Assistant", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# !!! UPDATE TO YOUR PATH !!!
SOURCE_DIRECTORY = "C:/SNCOA_Assistant/source_docs" 
CHROMA_PATH = "C:/SNCOA_Assistant/chroma_db_hybrid"

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage { border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; }
    
    /* TOOLTIP STYLES */
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
        width: 400px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        font-size: 0.85rem;
        font-weight: normal;
        
        /* Position the tooltip */
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -200px; /* Center it */
        
        /* Fade in */
        opacity: 0;
        transition: opacity 0.3s;
        
        /* Scroll if text is too long */
        max-height: 300px;
        overflow-y: auto;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

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

@st.cache_data(show_spinner=False)
def read_pdf_directly(filename):
    full_path = os.path.join(SOURCE_DIRECTORY, filename)
    if not os.path.exists(full_path):
        return f"[ERROR: File not found at {full_path}]"
    try:
        # Added a timer to debug log
        start = time.time() 
        loader = PyPDFLoader(full_path)
        pages = loader.load()
        text_content = "\n".join([p.page_content for p in pages])
        print(f"Loaded {filename} in {time.time() - start:.2f}s") 
        return text_content
    except Exception as e:
        return f"[ERROR Reading PDF: {e}]"

# --- ACRONYM DICTIONARY & EXPANSION ---
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

def expand_acronyms(text):
    for acronym, full_term in ACRONYM_MAP.items():
        pattern = r'\b' + re.escape(acronym) + r'\b'
        text = re.sub(pattern, f"{full_term} ({acronym})", text, flags=re.IGNORECASE)
    return text

def inject_citations_with_tooltips(text, docs):
    """
    Replaces [1], [2] with HTML tooltips containing the doc content.
    """
    def replace_match(match):
        citation_num = int(match.group(1)) # Get the number '1'
        doc_index = citation_num - 1       # List starts at 0
        
        if 0 <= doc_index < len(docs):
            source_text = docs[doc_index].page_content
            # Limit text length and escape HTML characters
            preview_text = html.escape(source_text[:600]) + ("..." if len(source_text) > 600 else "")
            source_name = docs[doc_index].metadata.get('source', 'Unknown').split('/')[-1]
            page_num = docs[doc_index].metadata.get('page', '?')
            
            # Return the HTML structure
            return (
                f'<div class="tooltip">[{citation_num}]'
                f'<span class="tooltiptext"><strong>{source_name} (Pg {page_num})</strong><br/><br/>{preview_text}</span>'
                f'</div>'
            )
        else:
            return match.group(0) # Return original [X] if no doc found

    # Regex to find [Number]
    return re.sub(r'\[(\d+)\]', replace_match, text)

# --- 6. MODEL HANDLERS ---
MODULE_FILE_MAP = {
    "Pre-Req": "Pre-Req_combined.pdf",
    "Module 0": "Module 0_Combined.pdf",
    "Module 1": "Module 1_combined.pdf",
    "Module 2": "Module 2_combined.pdf",
    "Module 3": "Module 3_combined.pdf",
    "Module 4": "Module4_Primer Strategic Focus Lab 01 Aug 2024.pdf"
}

CLOUD_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemma-3-27b-it", 
    "gemini-pro"
]

def get_llm_instance(engine_choice, model_name=None):
    if "Gemini" in engine_choice:
        target_model = model_name if model_name else "gemini-flash-latest"
        return ChatGoogleGenerativeAI(model=target_model, temperature=0)
    else:
        return ChatOllama(model="llama3.1", temperature=0)

# HANDLER: DIRECT READ GRADING
def grade_with_direct_read(rubric_filename, student_essay, prompt_template, content_scope, engine_choice):
    debug_log = [] 
    full_context_text = ""

    # Load Rubric
    rubric_text = read_pdf_directly(rubric_filename)
    full_context_text += f"\n--- SOURCE: {rubric_filename} (RUBRIC) ---\n{rubric_text}\n"
    debug_log.append(f"✅ Loaded Rubric")

    # Load Content
    if content_scope:
        for module in content_scope:
            target_file = MODULE_FILE_MAP.get(module, module)
            text = read_pdf_directly(target_file)
            full_context_text += f"\n--- SOURCE: {target_file} (CONTENT) ---\n{text}\n"
            debug_log.append(f"✅ Loaded {module}")

    # Load Style Guides
    style_guide = "AFSNCOA Style Guide August 2025.pdf"
    text = read_pdf_directly(style_guide)
    full_context_text += f"\n--- SOURCE: {style_guide} (STYLE) ---\n{text}\n"
    debug_log.append(f"✅ Loaded Style Guide")

    # Load T&Q only for Cloud
    tnq_file = "DAFH33-337 Tongue and Quill Dec 22.pdf"
    if "Gemini" in engine_choice:
        text = read_pdf_directly(tnq_file)
        full_context_text += f"\n--- SOURCE: {tnq_file} (STYLE) ---\n{text}\n"
        debug_log.append(f"✅ Loaded Tongue & Quill (Cloud Only)")
    else:
        debug_log.append(f"⚠️ Skipped Tongue & Quill (Too large for Local Mode)")

    # Retry Loop
    if "Gemini" in engine_choice:
        models_to_try = CLOUD_MODELS
    else:
        models_to_try = ["llama3.1"]

    last_error = None
    for model_name in models_to_try:
        try:
            debug_log.append(f"🔄 Attempting grade with: {model_name}...")
            llm = get_llm_instance(engine_choice, model_name)
            chain = prompt_template | llm
            stream = chain.stream({"context": full_context_text, "input": student_essay})
            debug_log.append(f"✅ Success! Streaming from: {model_name}")
            return stream, debug_log
        except Exception as e:
            debug_log.append(f"❌ {model_name} failed: {str(e)}")
            last_error = e
            continue
    raise last_error

# --- 7. UI SETUP ---
if "messages" not in st.session_state: st.session_state.messages = []
if "rubric_query" not in st.session_state: st.session_state.rubric_query = ""
if "last_evidence" not in st.session_state: st.session_state.last_evidence = []

with st.sidebar:
    st.header("Control Panel")
    engine_choice = st.radio(
        "Select AI Engine:", 
        ["☁️ Gemini (Google)", "🏠 Local (Ollama)"],
        help="Use 'Local' for unlimited grading. Requires 'ollama run llama3.1' in terminal."
    )
    st.markdown("---")
    mode = st.radio("Select Mission:", ["💬 Chat Assistant", "📝 Essay Grader"])
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.rubric_query = ""
        st.session_state.last_evidence = []
        st.rerun()
# === X-RAY Tool ===
    st.markdown("---")
    with st.expander("🔍 X-Ray: Inspect PDF Text"):
        st.caption("Check if the AI can actually read the text on a specific page.")
        
        # 1. Select File
        xray_file = st.selectbox("Select File:", list(MODULE_FILE_MAP.keys()))
        
        # 2. Select Page
        xray_page = st.number_input("Page Number (0-index):", min_value=0, value=0)
        
        # 3. Read Button
        if st.button("Read Page Text"):
            target_filename = MODULE_FILE_MAP[xray_file]
            full_path = os.path.join(SOURCE_DIRECTORY, target_filename)
            
            try:
                # Load just the specific page to save memory
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(full_path)
                pages = loader.load()
                
                if xray_page < len(pages):
                    raw_text = pages[xray_page].page_content
                    if not raw_text.strip():
                        st.error("⚠️ This page appears empty! It might be an image/chart.")
                    else:
                        st.text_area("Raw Text Output:", raw_text, height=300)
                else:
                    st.error(f"Page {xray_page} does not exist. (Max: {len(pages)-1})")
            except Exception as e:
                st.error(f"Error reading file: {e}")

st.title("✈️ SNCOA Instructor Assistant")
col_main, col_source = st.columns([0.65, 0.35], gap="medium")

# --- EVIDENCE BOARD (SIDEBAR) ---
with col_source:
    st.markdown("### 📖 Evidence Board")
    if mode == "💬 Chat Assistant":
        st.caption("Hover over citations [1] in chat to read sources.")
        # Optional: Still show list if desired, but minimizing for cleanliness
        if st.session_state.last_evidence:
             with st.expander("📚 Source List (Click to View)"):
                for i, doc in enumerate(st.session_state.last_evidence):
                    source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    st.text(f"[{i+1}] {source_name}")
    elif mode == "📝 Essay Grader":
        st.caption("Evidence display is minimized in Direct Read mode.")

# --- MAIN LOGIC ---
with col_main:
    # 1. CHAT ASSISTANT MODE
    if mode == "💬 Chat Assistant":
        # Display History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # NOTE: We do NOT use format_citations here anymore, 
                # because the HTML tooltips are already baked into the content.
                st.markdown(message["content"], unsafe_allow_html=True)  

        if prompt := st.chat_input("Ask about rubrics..."):
            
            # A. Expand Acronyms
            search_friendly_prompt = expand_acronyms(prompt)

            # B. Show User Message
            st.session_state.messages.append({"role": "user", "content": prompt}) 
            with st.chat_message("user"): st.markdown(prompt)
            
            if vectorstore:
                try:
                    # C. RETRIEVE DOCS MANUALLY (k=10 for better recall)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 40})
                    docs = retriever.invoke(search_friendly_prompt)
                    st.session_state.last_evidence = docs # Save for sidebar/debug
                    
                    # D. FORMAT CONTEXT FOR AI (This was missing in your code!)
                    formatted_context = ""
                    for i, doc in enumerate(docs):
                        source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                        page_num = doc.metadata.get('page', '0')
                        # We explicitly label [1], [2] so the AI sees them
                        formatted_context += f"\n--- SOURCE [{i+1}] ({source_name}, Pg {page_num}) ---\n{doc.page_content}\n"

                    # E. SYSTEM PROMPT
                    # 1. Convert Dictionary to a string list for the AI
                    acronyms_context = ", ".join([f"{k}={v}" for k, v in ACRONYM_MAP.items()])

                    system_prompt = (
                        "You are an SNCOA Instructor Assistant. Use the provided context to answer the user's question.\n"
                        f"KNOWN ACRONYMS: {acronyms_context}\n"  # <--- TEACH IT THE ACRONYMS
                        "If the user uses an acronym from this list (like CVF), treat it as synonymous with the full term in the text.\n"
                        "The context sources are labeled '--- SOURCE [X] ---'.\n"
                        "Synthesize the information from the provided sources. If an explicit definition is missing, infer it from the usage in the text.\n"
                        "Please cite sources as [1], [2], etc.\n\n"
                        "CONTEXT:\n{context}"
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt), 
                        ("human", "{input}")
                    ])

                    # F. GENERATE ANSWER
                    llm = get_llm_instance(engine_choice)
                    chain = prompt_template | llm
                    
                    # Use formatted_context here
                    response = chain.invoke({"context": formatted_context, "input": search_friendly_prompt})
                    
                    # G. PROCESS & INJECT TOOLTIPS
                    answer_text = response.content if hasattr(response, 'content') else str(response)
                    html_response = inject_citations_with_tooltips(answer_text, docs)
                    
                    # H. SAVE & DISPLAY
                    st.session_state.messages.append({"role": "assistant", "content": html_response})
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

    # 2. ESSAY GRADER MODE
    elif mode == "📝 Essay Grader":
        st.markdown("### Student Assessment")
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        def set_rubric(name): st.session_state.rubric_query = name
        with col_r1: st.button("📄 Mod 1", on_click=set_rubric, args=["Module 1 Personal Leadership Assessment Evaluation with Template 1 Jan 2025.pdf"])
        with col_r2: st.button("📋 Mod 2", on_click=set_rubric, args=["Module 2 Background Paper 01 May 2024.pdf"])
        with col_r3: st.button("🛡️ Mod 3", on_click=set_rubric, args=["Module 3 - 1 Strategic Environment Brief Instructions and Eval Instrument August  2025.pdf"])
        with col_r4: st.button("🎯 Refinement Strat", on_click=set_rubric, args=["Module 4 - AFSNCOA Personal Refinement Strategy 01 May 2024.pdf"])
        with col_r5: st.button("📚 Group Brief", on_click=set_rubric, args=["Module 4 - Plan of Action Briefing 01 May 2024.pdf"])
        
        assignment_filename = st.text_input("Selected Rubric File:", value=st.session_state.rubric_query, disabled=True)
        content_scope = st.multiselect("Include Content:", options=list(MODULE_FILE_MAP.keys()))
        student_essay = st.text_area("Paste Student Submission:", height=300)
        
        if st.button("Evaluate Submission", type="primary"):
            if not assignment_filename or not student_essay:
                st.warning("Please select a Rubric and paste the Essay.")
            else:
                # 1. DEFINE THE TEMPLATE
                system_prompt = (
                    "You are a strict Air Force Instructor. "
                    "Your job is to grade the student submission based on the FULL Rubrics, Content, and Style Guides provided."
                    "\n\n"
                    "*** STRICT OUTPUT RULES ***\n"
                    "You must NOT output generic text. You must output your evaluation using the EXACT Markdown format below. "
                    "Do not change the headers. Do not skip sections."
                    "\n\n"
                    "--- BEGIN OUTPUT TEMPLATE ---\n"
                    "# ✈️ ASSESSMENT REPORT\n\n"
                    "## 1. SOURCE ROLL CALL\n"
                    "(List exactly which files were used)\n\n"
                    "## 2. RUBRIC COMPLIANCE (Detailed Breakdown)\n"
                    "| Criteria | Status (Pass/Fail) | Deduction | Instructor Feedback |\n"
                    "| :--- | :--- | :--- | :--- |\n"
                    "| (Insert Criteria 1) | (Pass/Fail) | (-Points) | (Specific critique) |\n"
                    "| ... | ... | ... | ... |\n\n"
                    "## 3. CONTENT VERIFICATION (Fact-Check)\n"
                    "* **Claim 1:** (Quote student) -> **Verdict:** (Supported/Unsupported by [Module Name])\n\n"
                    "## 4. STYLE & FORMATTING\n"
                    "Review the essay for spelling, grammar, passive voice, and 'Tongue and Quill' violations.\n"
                    "* (List specific errors with cited text and line numbers or paragraph number if possible)\n\n"
                    "## 5. FINAL SCORE CALCULATION\n"
                    "**BASE SCORE:** 100\n"
                    "**TOTAL DEDUCTIONS:** (Sum of deductions)\n"
                    "### **FINAL SCORE:** (Calculated Score)/100\n"
                    "--- END OUTPUT TEMPLATE ---"
                    "\n\n"
                    "CONTEXT (Full Files Loaded from Disk):\n{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt), 
                    ("human", "STUDENT ESSAY:\n{input}")
                ])

                st.markdown("### Evaluation Result")
                result_container = st.empty()
                debug_expander = st.expander("🕵️ DEBUG: See Grading Process", expanded=False)

                try:
                    stream, debug_logs = grade_with_direct_read(
                        assignment_filename, 
                        student_essay, 
                        prompt_template,
                        content_scope,
                        engine_choice
                    )
                    
                    with debug_expander:
                        for log in debug_logs:
                            st.write(log)

                    response_text = result_container.write_stream(stream)
                    st.caption("✅ Assessment Complete.")
                        
                except Exception as e:
                    st.error(f"Grading Failed: {e}")