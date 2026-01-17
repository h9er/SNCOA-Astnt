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
    initial_sidebar_state="collapsed"
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
</style>
""", unsafe_allow_html=True)

CHROMA_PATH = "./chroma_db_hybrid"

# --- 4. ENGINE LOADING ---
@st.cache_resource
def load_vector_store():
    if not os.path.exists(CHROMA_PATH):
        return None
    # Use Local Ollama for Search (Free/Unlimited)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore

vectorstore = load_vector_store()

# --- 5. HELPER FUNCTIONS ---
def format_citations(text):
    pattern = r'\[(\d+)\]'
    return re.sub(pattern, r'<sup>\1</sup>', text)

# --- 6. SMART MODEL HANDLER ---
# The list of models to try, in order.
MODEL_FALLBACK_LIST = [
    "gemini-2.0-flash-lite-preview-02-05", # Fast & New
    "gemini-flash-latest",                  # Standard Pointer
    "gemma-3-27b-it",                       # Smart Backup
    "gemini-pro"                            # Legacy Backup
]

def get_response_with_fallback(input_text, retriever, prompt_template):
    """
    Tries models one by one. Updates the UI to show progress.
    """
    last_error = None
    status_placeholder = st.empty() # Create a temporary box on screen
    
    for model_name in MODEL_FALLBACK_LIST:
        try:
            status_placeholder.info(f"🔄 Attempting with model: **{model_name}**...")
            
            # Initialize Model (Zero Retries = Fail Fast)
            fallback_llm = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=0,
                max_retries=0 
            )
            
            # Build Chain
            chain = create_retrieval_chain(
                retriever, 
                create_stuff_documents_chain(fallback_llm, prompt_template)
            )
            
            # Run It
            response = chain.invoke({"input": input_text})
            
            # Success! Clear the status box
            status_placeholder.empty()
            return response, model_name

        except Exception as e:
            status_placeholder.warning(f"⚠️ {model_name} failed. Switching...")
            print(f"Server Log: {model_name} error: {e}")
            last_error = e
            time.sleep(0.5) # Small pause so user sees the switch
            continue 

    status_placeholder.error("❌ All models failed. Check API Key quota.")
    raise last_error

# --- 7. SESSION STATE SETUP ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []

# --- 8. MAIN INTERFACE ---
st.title("✈️ SNCOA Instructor Assistant")

col_chat, col_source = st.columns([0.65, 0.35], gap="medium")

# --- RIGHT COLUMN: SOURCES ---
with col_source:
    st.markdown("### 📖 Source Reference")
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
        st.caption("Sources will appear here automatically.")

# --- LEFT COLUMN: CHAT ---
with col_chat:
    # 1. Render History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(format_citations(message["content"]), unsafe_allow_html=True)

    # 2. Handle Input
    if prompt := st.chat_input("Ask about rubrics, guidance, or course concepts..."):
        
        # Append User Message Immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Generate Answer
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            system_prompt = (
                "You are an expert Air Force Instructor Assistant. "
                "Use the provided context to answer. "
                "\n\n"
                "CITATION RULES:\n"
                "1. Cite sources using brackets like [1].\n"
                "2. Do not list sources at the end.\n"
                "\n\n"
                "CONTEXT:\n"
                "{context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Use the Fallback Function
            try:
                response, used_model = get_response_with_fallback(
                    prompt, 
                    retriever, 
                    prompt_template
                )
                
                raw_answer = response["answer"]
                sources = response["context"]

                # 4. CRITICAL: Save everything to State BEFORE Rerunning
                st.session_state.current_sources = sources
                
                debug_footer = f"\n\n_<small style='color:grey'>Generated by: {used_model}</small>_"
                final_text = raw_answer + debug_footer
                
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
                # 5. Force Refresh (Updates the Right Column)
                st.rerun()
                
            except Exception as e:
                st.error(f"Application Error: {e}")
        else:
            st.error("Database not connected. Run ingest script.")