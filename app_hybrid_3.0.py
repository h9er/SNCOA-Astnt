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
from langchain_core.documents import Document

# --- 1. COMPATIBILITY FIX ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

load_dotenv()

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SNCOA Instructor Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
    }
    sup {
        color: #d90429;
        font-weight: bold;
        cursor: help;
    }
    [data-testid="column"]:nth-of-type(2) {
        background-color: rgba(248, 249, 250, 0.05); 
        border-left: 1px solid #ddd;
        padding: 20px;
        height: 100vh;
        overflow-y: auto;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    div.stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

CHROMA_PATH = "./chroma_db_hybrid"

# --- 4. ENGINE LOADING ---
@st.cache_resource
def load_vector_store():
    if not os.path.exists(CHROMA_PATH):
        return None
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore

vectorstore = load_vector_store()

# --- 5. HELPER FUNCTIONS ---
def format_citations(text):
    pattern = r'\[(\d+)\]'
    return re.sub(pattern, r'<sup>\1</sup>', text)

# --- 6. SMART MODEL HANDLERS ---
MODEL_FALLBACK_LIST = [
    "gemini-2.0-flash-lite-preview-02-05", 
    "gemini-flash-latest",
    "gemma-3-27b-it", 
    "gemini-pro"
]

# GLOBAL MAP: Defines the Content Files (Readings)
# Based on your inventory scan
MODULE_FILE_MAP = {
    "Pre-Req": "Pre-Req_combined.pdf",
    "Module 0": "Module 0_Combined.pdf",
    "Module 1": "Module 1_combined.pdf",
    "Module 2": "Module 2_combined.pdf",
    "Module 3": "Module 3_combined.pdf",
    "Module 4": "Module4_Primer Strategic Focus Lab 01 Aug 2024.pdf"
}

# HELPER: PRECISION HUNTER
def get_specific_file(retriever, target_filename, description):
    """
    Bypasses similarity search completely.
    Scans the database inventory to find ALL chunks belonging to the target file.
    Guarantees 100% of the document is retrieved.
    """
    vs = retriever.vectorstore 
    
    # 1. Fetch Metadata for the ENTIRE database (Fast operation)
    # We need to find which IDs belong to our target file
    try:
        # Get all metadata to filter by filename
        db_data = vs.get(include=['metadatas'])
        
        target_ids = []
        # Loop through all database entries to find our file
        for i, meta in enumerate(db_data['metadatas']):
            # Check if our target filename is in the source path
            if target_filename.lower() in meta.get('source', '').lower():
                target_ids.append(db_data['ids'][i])
        
        if not target_ids:
            return [], [f"⚠️ {description}: File '{target_filename}' not found in database inventory."]
            
        # 2. Fetch the actual content for those IDs
        # Now we grab the text for exactly those chunks
        content_data = vs.get(ids=target_ids, include=['documents', 'metadatas'])
        
        # 3. Reconstruct as LangChain Documents
        docs = []
        for i, text in enumerate(content_data['documents']):
            # Create a Document object
            doc = Document(page_content=text, metadata=content_data['metadatas'][i])
            docs.append(doc)
            
        return docs, [f"✅ {description}: Retrieved full document ({len(docs)} chunks)"]

    except Exception as e:
        return [], [f"❌ {description}: Retrieval Error - {str(e)}"]

# HANDLER 1: FOR CHAT
def get_chat_response(input_text, retriever, prompt_template):
    last_error = None
    status_placeholder = st.empty()
    
    for model_name in MODEL_FALLBACK_LIST:
        try:
            status_placeholder.info(f"🔄 Chatting with model: **{model_name}**...")
            fallback_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=0)
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(fallback_llm, prompt_template))
            response = chain.invoke({"input": input_text})
            status_placeholder.empty()
            return response, model_name
        except Exception as e:
            status_placeholder.warning(f"⚠️ {model_name} failed. Switching...")
            time.sleep(0.5)
            last_error = e
            continue 
    status_placeholder.error("❌ All models failed.")
    raise last_error

# HANDLER 2: FOR GRADING (Using Precision Hunter)
def grade_with_fallback(rubric_topic, student_essay, retriever, prompt_template, content_scope):
    last_error = None
    status_placeholder = st.empty()
    debug_log = [] 
    
    # --- STEP 1: RETRIEVE THE RUBRIC ---
    status_placeholder.info(f"🔎 Hunting for Rubric file...")
    docs_rubric, logs = get_specific_file(retriever, rubric_topic, "RUBRIC")
    debug_log.extend(logs)
    
    if not docs_rubric:
        debug_log.append(f"⚠️ Rubric fallback: searching generically for '{rubric_topic}'")
        docs_rubric = retriever.invoke(rubric_topic)

    # --- STEP 2: RETRIEVE SCOPED CONTENT ---
    docs_content = []
    if content_scope:
        status_placeholder.info(f"🔎 Pulling content for: {', '.join(content_scope)}...")
        for module in content_scope:
            target_filename = MODULE_FILE_MAP.get(module, module)
            found_docs, logs = get_specific_file(retriever, target_filename, f"CONTENT ({module})")
            docs_content.extend(found_docs)
            debug_log.extend(logs)
    else:
        guess_query = rubric_topic.replace("Rubric", "").strip()
        docs_content = retriever.invoke(guess_query)

    # --- STEP 3: RETRIEVE STYLE GUIDES (EXACT NAMES) ---
    status_placeholder.info("🔎 Checking Style Guides...")
    
    # Updated to match your inventory exactly
    docs_style, logs_style = get_specific_file(retriever, "AFSNCOA Style Guide August 2025.pdf", "STYLE")
    docs_quill, logs_quill = get_specific_file(retriever, "DAFH33-337 Tongue and Quill Dec 22.pdf", "QUILL")
    
    debug_log.extend(logs_style)
    debug_log.extend(logs_quill)
    
    # Fallback
    if not docs_style: docs_style = retriever.invoke("SNCOA Style Guide")
    if not docs_quill: docs_quill = retriever.invoke("Tongue and Quill")

    # --- VISIBILITY ---
    with st.expander("🕵️ DEBUG: See Retrieved Documents"):
        for entry in debug_log:
            st.write(entry)
            
    # --- STEP 4: MANUAL CONTEXT BUILDER ---
    all_docs = docs_rubric + docs_content + docs_style + docs_quill
    
    # Deduplicate
    unique_docs = []
    seen = set()
    for d in all_docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)
    
    if not unique_docs:
         status_placeholder.warning("⚠️ Low context. Grading may be inaccurate.")
         
    formatted_context = ""
    for doc in unique_docs:
        filename = doc.metadata.get('source', 'Unknown').split(os.sep)[-1]
        formatted_context += f"\n--- SOURCE: {filename} ---\n{doc.page_content}\n"

    # --- STEP 5: GENERATE GRADE ---
    for model_name in MODEL_FALLBACK_LIST:
        try:
            status_placeholder.info(f"📝 Grading with model: **{model_name}**...")
            fallback_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=0)
            
            chain = prompt_template | fallback_llm
            response = chain.invoke({"context": formatted_context, "input": student_essay})
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            status_placeholder.empty()
            return response_text, unique_docs, model_name

        except Exception as e:
            status_placeholder.warning(f"⚠️ {model_name} failed. Switching...")
            time.sleep(0.5)
            last_error = e
            continue 
            
    status_placeholder.error("❌ All models failed.")
    raise last_error

# --- 7. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "rubric_query" not in st.session_state:
    st.session_state.rubric_query = ""

# --- 8. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Select Mission:", ["💬 Chat Assistant", "📝 Essay Grader"])
    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.current_sources = []
        st.session_state.rubric_query = ""
        st.rerun()

# --- 9. MAIN LAYOUT ---
st.title("✈️ SNCOA Instructor Assistant")
col_main, col_source = st.columns([0.65, 0.35], gap="medium")

# --- RIGHT COLUMN (EVIDENCE) ---
with col_source:
    st.markdown("### 📖 Evidence Board")
    st.markdown("---")
    if st.session_state.current_sources:
        for i, doc in enumerate(st.session_state.current_sources):
            source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
            page_num = doc.metadata.get('page', '?')
            with st.container():
                st.markdown(f"**Reference [{i+1}]** — *{source_name}* (Pg. {page_num})")
                st.info(doc.page_content) 
                st.markdown("---")
    else:
        st.caption("Relevant rubrics and regs will appear here.")

# --- LEFT COLUMN (ACTION) ---
with col_main:
    
    # === MODE 1: CHAT ASSISTANT ===
    if mode == "💬 Chat Assistant":
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(format_citations(message["content"]), unsafe_allow_html=True)

        if prompt := st.chat_input("Ask about rubrics, guidance, or course concepts..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                system_prompt = (
                    "You are an expert Air Force Instructor Assistant. "
                    "Use the provided context to answer, review it line by line do not skip anything. "
                    "Be concise and precise. "
                    "Be inquisitive if the question is ambiguous to obtain clarity and provide the best answer. "
                    "Cite sources using brackets like [1]."
                    "\n\nCONTEXT:\n{context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt), ("human", "{input}")
                ])

                try:
                    response, used_model = get_chat_response(prompt, retriever, prompt_template)
                    st.session_state.current_sources = response["context"]
                    debug_footer = f"\n\n_<small style='color:grey'>Generated by: {used_model}</small>_"
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"] + debug_footer})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # === MODE 2: ESSAY GRADER ===
    elif mode == "📝 Essay Grader":
        st.markdown("### Student Assessment")
        
        # --- 1. RUBRIC SELECTION (UPDATED FILENAMES) ---
        st.write("Select a Rubric:")
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        def set_rubric(name): st.session_state.rubric_query = name
        
        # NOTE: Using EXACT filenames from your inventory!
        with col_r1: st.button("📄 Mod 1", on_click=set_rubric, args=["Module 1 Personal Leadership Assessment Evaluation with Template 1 Jan 2025.pdf"])
        with col_r2: st.button("📋 Mod 2", on_click=set_rubric, args=["Module 2 Background Paper 01 May 2024.pdf"])
        with col_r3: st.button("🛡️ Mod 3", on_click=set_rubric, args=["Module 3 - 1 Strategic Environment Brief Instructions and Eval Instrument August  2025.pdf"])
        with col_r4: st.button("🎯  Mod 4 - Ref Strat", on_click=set_rubric, args=["Module 4 - AFSNCOA Personal Refinement Strategy 01 May 2024.pdf"])
        with col_r5: st.button("📋 Mod 4 - Plan of Action", on_click=set_rubric, args=["Module 4 - Plan of Action Briefing 01 May 2024.pdf"])

        # --- 2. INPUTS ---
        st.info("Paste the student's work below.")
        
        assignment_name = st.text_input("Assignment Topic (Rubric Search):", value=st.session_state.rubric_query)
        
        # --- 3. CONTENT SCOPE ---
        content_scope = st.multiselect(
            "Include Content/Readings from:",
            options=list(MODULE_FILE_MAP.keys()), 
            help="Select modules to fact-check the student's claims."
        )
        
        student_essay = st.text_area("Paste Student Submission:", height=300)
        
        if st.button("Evaluate Submission", type="primary"):
            if not assignment_name or not student_essay:
                st.warning("Please provide both the Assignment Topic and the Student Essay.")
            elif vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                # UPDATED: Force-Table Prompt
                system_prompt = (
                    "You are a strict Air Force Instructor. "
                    "Your job is to grade the student submission based on the Rubrics, Course Content, and Style Guides."
                    "\n\n"
                    "INSTRUCTIONS:\n"
                    "0. **SOURCE ROLL CALL:** List exactly which files you are reading.\n"
                    "1. **RUBRIC EXTRACTION (CRITICAL):** Before grading, you must internally review the full rubric. Create a Markdown Table in your response listing EVERY grading criteria found in the rubric files (Part I, Part II, etc.) and the student's status for each.\n"
                    "2. **NO ESTIMATION:** You are forbidden from 'estimating' scores. If a rubric section is missing from your context, state 'Criteria Not Found' rather than guessing.\n"
                    "3. **CONTENT CHECK:** Verify the student's concepts against the loaded Module content.\n"
                    "4. **REFERENCES:** Check against 'Tongue and Quill'.\n"
                    "5. **FINAL SCORE:** Calculate the mathematical sum based *only* on the table you created."
                    "\n\n"
                    "CONTEXT (Full Files):\n"
                    "{context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt), ("human", "STUDENT ESSAY:\n{input}"),
                ])

                try:
                    with st.spinner("Retrieving Rubric, Content & Grading..."):
                        response_text, docs, used_model = grade_with_fallback(
                            assignment_name, 
                            student_essay, 
                            retriever, 
                            prompt_template,
                            content_scope
                        )
                        st.session_state.current_sources = docs
                        st.markdown("### Evaluation Result")
                        st.markdown(format_citations(response_text), unsafe_allow_html=True)
                        st.caption(f"Graded by: {used_model}")
                        
                except Exception as e:
                    st.error(f"Grading Failed: {e}")