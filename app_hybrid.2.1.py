import os
import sys
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
        background-color: #f8f9fa;
        border-left: 1px solid #ddd;
        padding: 20px;
        height: 100vh;
        overflow-y: auto;
    }
    /* Grader Mode specific styles */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        background-color: #f0f2f6;
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

# HANDLER 1: FOR CHAT (Standard RAG)
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

# HANDLER 2: FOR GRADING (Sniper Search)
def grade_with_fallback(rubric_topic, student_essay, retriever, prompt_template):
    """
    1. Search Database using ONLY 'rubric_topic'
    2. Feed found Docs + 'student_essay' to LLM
    """
    last_error = None
    status_placeholder = st.empty()
    
    # STEP 1: RETRIEVE DOCUMENTS MANUALLY (The Sniper Shot)
    status_placeholder.info(f"🔎 Searching database for: '{rubric_topic}'...")
    relevant_docs = retriever.invoke(rubric_topic)
    
    if not relevant_docs:
        raise ValueError("No documents found. Try a different Assignment Name.")

    # STEP 2: GENERATE GRADE WITH FALLBACK
    for model_name in MODEL_FALLBACK_LIST:
        try:
            status_placeholder.info(f"📝 Grading with model: **{model_name}**...")
            
            fallback_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=0)
            
            # Note: We use 'create_stuff_documents_chain' directly, bypassing the retrieval chain
            # because we already have the docs from Step 1.
            chain = create_stuff_documents_chain(fallback_llm, prompt_template)
            
            # We pass the Docs (context) and the Essay (input) separately
            response_text = chain.invoke({
                "context": relevant_docs,
                "input": student_essay
            })
            
            status_placeholder.empty()
            return response_text, relevant_docs, model_name

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

# --- 8. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Select Mission:", ["💬 Chat Assistant", "📝 Essay Grader"])
    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.current_sources = []
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

    # === MODE 2: ESSAY GRADER (UPDATED LOGIC) ===
    elif mode == "📝 Rubric Grader":
        st.markdown("### Student Assessment")
        
        # --- NEW: RUBRIC PRESET BUTTONS ---
        st.markdown("Select a Rubric:")
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        
        # Initialize the search term in session state if missing
        if "rubric_query" not in st.session_state:
            st.session_state.rubric_query = ""

        # Define Buttons - Click updates the session state
        with col_r1:
            if st.button("📄 Module 1"):
                st.session_state.rubric_query = "Module 1 Personal Leadership Assessment Evaluation with Template 1 Jan 2025"
        with col_r2:
            if st.button("📄 Module 2"):
                st.session_state.rubric_query = "Module 2 Background Paper 01 May 2024"
        with col_r3:
            if st.button("📋 Module 3 Strat Brief"):
                st.session_state.rubric_query = "Module 3 - 1 Strategic Environment Brief Instructions and Eval Instrument August  2025"
        with col_r4:
            if st.button("📋 Module 4 Refinement Strategy"):
                st.session_state.rubric_query = "Module 4 - AFSNCOA Personal Refinement Strategy 01 May 2024"
        with col_r5:
            if st.button("📋 Module 4 Strat Brief"):
                st.session_state.rubric_query = "Module 4 - Plan of Action Briefing 01 May 2024"

        # --- INPUTS ---
        st.info("Paste the student's work below.")
        
        # The Input Box is now bound to the session state variable
        assignment_name = st.text_input(
            "Assignment Topic (Rubric Search):", 
            value=st.session_state.rubric_query,
            key="rubric_input_box" # Unique ID
        )
        
        student_essay = st.text_area("Paste Student Submission:", height=300)
        
        if st.button("Evaluate Submission", type="primary"):
            if not assignment_name or not student_essay:
                st.warning("Please provide both the Assignment Topic and the Student Essay.")
            elif vectorstore:
                # 1. Setup Retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                
                # 2. Setup Prompt
                system_prompt = (
                    "You are a strict Air Force Instructor. "
                    "Your job is to grade the student submission based ONLY on the provided rubrics/context. "
                    "Be critical. Identify specific areas where the student met or failed the standard. "
                    "Cite the rubric for every critique."
                    "\n\n"
                    "CONTEXT (RUBRICS & REGS):\n"
                    "{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "STUDENT ESSAY:\n{input}"),
                ])

                try:
                    with st.spinner("Retrieving Rubric & Grading..."):
                        # CALL THE SNIPER FUNCTION
                        response_text, docs, used_model = grade_with_fallback(
                            assignment_name, 
                            student_essay, 
                            retriever, 
                            prompt_template
                        )
                        
                        # Save sources
                        st.session_state.current_sources = docs
                        
                        st.markdown("### Evaluation Result")
                        st.markdown(format_citations(response_text), unsafe_allow_html=True)
                        st.caption(f"Graded by: {used_model}")
                        
                except Exception as e:
                    st.error(f"Grading Failed: {e}")