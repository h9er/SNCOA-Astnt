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
    /* Chat Bubbles */
    .stChatMessage {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
    }
    /* Citations */
    sup {
        color: #d90429;
        font-weight: bold;
        cursor: help;
    }
    /* Right Column (Sources) */
    [data-testid="column"]:nth-of-type(2) {
        /* We use a transparent background so it adapts to dark/light mode automatically */
        background-color: rgba(248, 249, 250, 0.05); 
        border-left: 1px solid #ddd;
        padding: 20px;
        height: 100vh;
        overflow-y: auto;
    }
    /* Grader Text Area - THE FIX */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        /* Removed 'background-color' so it adapts to your Dark Mode settings */
        /* Removed specific text color so it stays white in Dark Mode and black in Light Mode */
    }
    /* Buttons */
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

# GLOBAL MAP: Defines both the Dropdown Options AND the Filenames
# Update the filenames on the right to match your actual files exactly.
MODULE_FILE_MAP = {
    "Pre-Req": "Pre-Req_combined.pdf",
    "Module 0": "Module 0_Combined.pdf",
    "Module 1": "Module 1_combined.pdf",
    "Module 2": "Module 2_combined.pdf",
    "Module 3": "Module 3_combined.pdf",
    "Module 4": "Module4_Primer Strategic Focus Lab 01 Aug 2024.pdf"
}

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

# HANDLER 2: FOR GRADING (With Smart/Robust Filtering)
def grade_with_fallback(rubric_topic, student_essay, retriever, prompt_template, content_scope):
    last_error = None
    status_placeholder = st.empty()
    debug_log = [] 
    
    # NOTE: Uses GLOBAL 'MODULE_FILE_MAP'
    
    # --- STEP 1: RETRIEVE THE RUBRIC ---
    status_placeholder.info(f"🔎 Hunting for Rubric: '{rubric_topic}'...")
    docs_rubric = retriever.invoke(rubric_topic)
    debug_log.extend([f"RUBRIC: {d.metadata.get('source', 'Unknown').split(os.sep)[-1]}" for d in docs_rubric])
    
    # --- STEP 2: RETRIEVE SCOPED CONTENT (Smart Filter) ---
    docs_content = []
    if content_scope:
        status_placeholder.info(f"🔎 Pulling content for: {', '.join(content_scope)}...")
        
        # Access the raw database engine for broader search
        vs = retriever.vectorstore 
        
        for module in content_scope:
            target_filename = MODULE_FILE_MAP.get(module, module)
            
            # 1. Broad Search (Get 15 candidates)
            raw_docs = vs.similarity_search(target_filename, k=15)
            
            # 2. Smart Filter (Case-Insensitive Check)
            filtered_docs = []
            for d in raw_docs:
                source_path = d.metadata.get('source', '')
                # Normalize both to lowercase to ignore "Combined" vs "combined" issues
                if target_filename.lower() in source_path.lower():
                    filtered_docs.append(d)
                else:
                    # Log what was rejected so we can debug it
                    rejected_name = source_path.split(os.sep)[-1]
                    debug_log.append(f"⚠️ REJECTED: '{rejected_name}' (Did not match '{target_filename}')")

            # 3. Fallback: If filter killed everything, take the top 2 anyway
            if not filtered_docs:
                debug_log.append(f"⚠️ Strict filter returned 0 for {module}. Forced fallback to top 2 results.")
                docs_content.extend(raw_docs[:2]) 
            else:
                docs_content.extend(filtered_docs)
                for d in filtered_docs:
                     src = d.metadata.get('source', 'Unknown').split(os.sep)[-1]
                     debug_log.append(f"✅ CONTENT ({module}): {src}")

    else:
        # Fallback if no scope selected
        guess_query = rubric_topic.replace("Rubric", "").strip()
        docs_content = retriever.invoke(guess_query)

    # Remove duplicates (in case of overlap)
    unique_content = []
    seen_content = set()
    for d in docs_content:
        # Use content + source to identify uniqueness
        unique_id = d.page_content + d.metadata.get('source', '')
        if unique_id not in seen_content:
            unique_content.append(d)
            seen_content.add(unique_id)
    docs_content = unique_content

    # --- STEP 3: RETRIEVE STYLE GUIDES ---
    docs_style = retriever.invoke("AFSNCOA Style Guide August 2025.pdf")
    docs_quill = retriever.invoke("DAFH33-337 Tongue and Quill Dec 22.pdf")
    debug_log.extend([f"STYLE: {d.metadata.get('source', 'Unknown').split(os.sep)[-1]}" for d in docs_style])
    
    # --- VISIBILITY ---
    with st.expander("🕵️ DEBUG: See Retrieved Documents"):
        # Show the log cleanly
        for entry in debug_log:
            st.write(entry)
            
    # --- STEP 4: MANUAL CONTEXT BUILDER ---
    all_docs = docs_rubric + docs_content + docs_style + docs_quill
    
    if not all_docs:
         status_placeholder.warning("⚠️ Low context. Grading may be inaccurate.")
         
    formatted_context = ""
    for doc in all_docs:
        filename = doc.metadata.get('source', 'Unknown').split(os.sep)[-1]
        formatted_context += f"\n--- SOURCE: {filename} ---\n{doc.page_content}\n"

    # --- STEP 5: GENERATE GRADE ---
    for model_name in MODEL_FALLBACK_LIST:
        try:
            status_placeholder.info(f"📝 Grading with model: **{model_name}**...")
            fallback_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=0)
            
            chain = prompt_template | fallback_llm
            
            response = chain.invoke({
                "context": formatted_context, 
                "input": student_essay
            })
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            status_placeholder.empty()
            return response_text, all_docs, model_name

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
        # Render History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(format_citations(message["content"]), unsafe_allow_html=True)

        # Input
        if prompt := st.chat_input("Ask about rubrics, guidance, or course concepts..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                system_prompt = (
                    "You are an expert Air Force Instructor Assistant. "
                    "Use the provided context to answer. "
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
        
        # --- 1. RUBRIC SELECTION ---
        st.write("Select a Rubric:")
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        def set_rubric(name): st.session_state.rubric_query = name
        
        # Remember to update these args to match your ACTUAL rubric filenames!
        with col_r1: st.button("📄 Module 1", on_click=set_rubric, args=["Module 1 Personal Leadership Assessment Evaluation with Template 1 Jan 2025.pdf"])
        with col_r2: st.button("📄 Module 2", on_click=set_rubric, args=["Module 2 Background Paper 01 May 2024.pdf"])
        with col_r3: st.button("📋 Module 3", on_click=set_rubric, args=["Module 3 - 1 Strategic Environment Brief Instructions and Eval Instrument August  2025.pdf"])
        with col_r4: st.button("📄 Module 4 - Ref Strat", on_click=set_rubric, args=["Module 4 - AFSNCOA Personal Refinement Strategy 01 May 2024.pdf"])
        with col_r5: st.button("📄 Module 4 - Plan of Action", on_click=set_rubric, args=["Module 4 - Plan of Action Briefing 01 May 2024.pdf"])
        
        # --- 2. INPUTS ---
        st.info("Paste the student's work below.")
        
        assignment_name = st.text_input("Assignment Topic (Rubric Search):", value=st.session_state.rubric_query)
        # --- 3. CONTENT SCOPE (NEW) ---
        content_scope = st.multiselect(
            "Include Content/Readings from:",
            # OLD HARDCODED LIST: ["Module 1", "Module 2", ...] -> DELETE THIS
            # NEW DYNAMIC LIST:
            options=list(MODULE_FILE_MAP.keys()), 
            help="Select modules to fact-check the student's claims."
        )
        
        student_essay = st.text_area("Paste Student Submission:", height=300)
        
        if st.button("Evaluate Submission", type="primary"):
            if not assignment_name or not student_essay:
                st.warning("Please provide both the Assignment Topic and the Student Essay.")
            elif vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                
# UPDATED: Prompt with Source Roll Call
                system_prompt = (
                    "You are a strict Air Force Instructor. "
                    "Your job is to grade the student submission based on the Rubrics, Course Content, and Style Guides."
                    "\n\n"
                    "INSTRUCTIONS:\n"
                    "0. **SOURCE ROLL CALL:** At the very top, list the specific documents you are using to grade this assignment (e.g., 'Rubric: Module 1', 'Content: Module 1_combined.pdf').\n"
                    "1. **SCORE:** Start with a **SUGGESTED SCORE** (e.g., 'Score: 85/100'). Deduct points for rubric failures.\n"
                    "2. **CONTENT CHECK:** Verify that the student's ideas actually come from the Source Material provided in the context.\n"
                    "3. **STYLE GUIDE:** Ensure adherence to the 'AFSNCOA Style Guide August 2025.pdf' and the 'DAFH33-337 Tongue and Quill Dec 22.pdf'.\n"
                    "4. **REFERENCES:** Cross-check citations against the 'AFSNCOA Style Guide August 2025.pdf' and 'DAFH33-337 Tongue and Quill Dec 22.pdf'.\n"
                    "5. **TEMPLATE:** Treat the template as a flexible guide, not a rigid law.\n"
                    "6. **CITATIONS:** Cite the specific document and student input for every critique.\n"
                    "7. **FEEDBACK:** Provide constructive feedback for improvement.\n"
                    "8. **FORMAT:** Use associated rubric table and follow the table explicitly (e.g., 'Module 1' Part II, Student included 3/4 required strengths receiving 21/25 points per rubric).\n"
                    "\n\n"
                    "CONTEXT (Rubrics, Readings, & Guides):\n"
                    "{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt), ("human", "STUDENT ESSAY:\n{input}"),
                ])

                try:
                    with st.spinner("Retrieving Rubric, Content & Grading..."):
                        # We pass 'content_scope' to the function here
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