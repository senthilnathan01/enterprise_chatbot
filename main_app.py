# main_app.py
"""
Main Streamlit application file for the Multi-Modal Q&A Chatbot.
Handles UI, state management, and orchestrates calls to other modules.
"""

import streamlit as st
import google.generativeai as genai
import chromadb
import os
import time
import random
import string

# Import functions from our modules
from config import (
    DEFAULT_COLLECTION_NAME_PREFIX, ALL_SUPPORTED_TYPES,
    VECTOR_DB_PERSIST_PATH
)
from utils import log_message, generate_unique_id
from vector_store import get_embedding_function, process_and_embed, get_relevant_context
from qa_engine import is_data_query, answer_data_query, generate_answer_from_context, generate_followup_questions

# --- Page Configuration ---
st.set_page_config(
    page_title="Conversational Multi-Modal Q&A",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar for history/logs initially
)

st.title("📄 Conversational Multi-Modal Q&A")
st.caption("Chat, Upload Documents & Ask Questions (Supports Text, Data, Images, Links, OCR)")

# --- Global Variables / Initializations ---
# (These should ideally be accessed via function calls or session state,
# but for simplicity in the callback, we might reference them if initialized early)
# Initialized later using @st.cache_resource

# --- API Key and Model Configuration ---
# (Same as before - use environment variables or secrets.toml)
try:
    google_api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.sidebar.warning("Google API Key not found.", icon="⚠️") # Show in sidebar now
    google_api_key = st.sidebar.text_input("Enter Google AI API Key:", type="password", key="api_key_input_main") # Keep in sidebar

if not google_api_key:
    st.info("Please provide your Google AI API Key in the sidebar to proceed.")
    st.stop()

try:
    genai.configure(api_key=google_api_key)
    # Use session state to track configuration success
    if "api_configured" not in st.session_state:
        log_message("Google AI API configured successfully.", "info")
        st.session_state.api_configured = True # Mark as configured
except Exception as e:
    st.error(f"Fatal Error: Could not configure Google AI API: {e}")
    log_message(f"Fatal Error: Could not configure Google AI API: {e}", "error")
    st.stop()

# --- Initialize ChromaDB Client and Embedding Function ---
# Use @st.cache_resource to ensure these are initialized once per session
@st.cache_resource
def initialize_chroma_client():
    # This function will only run once per session
    if "chroma_client_initialized" not in st.session_state:
         log_message("Initializing ChromaDB client...", "debug")
         if VECTOR_DB_PERSIST_PATH:
              log_message(f"Using persistent ChromaDB storage at: {VECTOR_DB_PERSIST_PATH}", "info")
              client = chromadb.PersistentClient(path=VECTOR_DB_PERSIST_PATH)
         else:
              log_message("Using in-memory ChromaDB storage.", "info")
              client = chromadb.Client()
         st.session_state.chroma_client_initialized = True
         return client
    # If already initialized, it should be available via the cache mechanism indirectly
    # This part might be tricky, accessing the cached resource directly might be needed
    # For simplicity, let's assume it's available after first run.

@st.cache_resource
def initialize_embedding_function(api_key):
     if "embedding_func_initialized" not in st.session_state:
          log_message("Initializing embedding function...", "debug")
          func = get_embedding_function(api_key)
          if func:
               st.session_state.embedding_func_initialized = True
               return func
          else:
               st.error("Fatal Error: Could not initialize embedding function.")
               st.stop()

# Call initialization functions (cached)
chroma_client = initialize_chroma_client()
embedding_function = initialize_embedding_function(google_api_key)

# --- Session State Management ---
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        # Initialize log here if it's the very first run
        if "processing_log" not in st.session_state: st.session_state.processing_log = []
        log_message(f"New session started: {st.session_state.session_id}", "info")
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = DEFAULT_COLLECTION_NAME_PREFIX + st.session_state.session_id
    # Other state variables are initialized as needed or checked before use
    if "messages" not in st.session_state: st.session_state.messages = []
    if "processed_files" not in st.session_state: st.session_state.processed_files = {}
    if "chroma_collection" not in st.session_state: st.session_state.chroma_collection = None
    if "data_frames" not in st.session_state: st.session_state.data_frames = {}
    if "crawled_data" not in st.session_state: st.session_state.crawled_data = {}
    if "rerun_query" not in st.session_state: st.session_state.rerun_query = None
    if "processing_status" not in st.session_state: st.session_state.processing_status = "idle" # idle, processing, success, error

initialize_session_state()

# --- File Processing Callback ---
def handle_file_upload():
    """Callback function triggered when files are uploaded."""
    st.session_state.processing_status = "processing"
    uploaded_files = st.session_state.get("doc_uploader_key", []) # Get files from uploader's key

    if not uploaded_files:
        log_message("File uploader changed, but no files found.", "warning")
        st.session_state.processing_status = "idle"
        return

    log_message(f"File upload detected ({len(uploaded_files)} files). Starting processing...", "info")

    # --- Reset state for this processing run ---
    # Don't clear chat messages here unless intended
    st.session_state.processed_files = {} # Reset status for this batch
    st.session_state.data_frames = {} # Clear previous dataframes
    st.session_state.crawled_data = {} # Clear previous crawled data
    # Keep logs from previous runs? Or clear? Let's clear for this job.
    st.session_state.processing_log = [f"[{time.strftime('%H:%M:%S')}][INFO] Starting processing for {len(uploaded_files)} files..."]

    # --- Get Chroma Client and Embedding Function ---
    # They should be available via the cached resources
    client = initialize_chroma_client() # Re-access cached client
    emb_func = initialize_embedding_function(google_api_key) # Re-access cached function

    if not client or not emb_func:
         log_message("Cannot process files: Chroma client or embedding function not available.", "error")
         st.session_state.processing_status = "error"
         return

    # --- Delete and recreate Chroma collection ---
    collection_name = st.session_state.collection_name
    try:
        log_message(f"Attempting to delete old collection: '{collection_name}'.", "info")
        client.delete_collection(name=collection_name)
        log_message(f"Collection '{collection_name}' deleted.", "info")
    except Exception as e:
        log_message(f"Collection '{collection_name}' not found or could not be deleted: {e}", "warning")

    # Create the new collection
    try:
         collection = client.get_or_create_collection(
             name=collection_name,
             embedding_function=emb_func,
             metadata={"hnsw:space": "cosine"}
         )
         st.session_state.chroma_collection = collection # Store collection object in state
         log_message(f"Created/recreated collection: '{collection_name}'.", "info")
    except Exception as e:
         log_message(f"Fatal Error: Failed to create ChromaDB collection: {e}", "error")
         st.session_state.processing_status = "error"
         st.session_state.chroma_collection = None
         return # Stop if collection fails

    # --- Process Files ---
    if st.session_state.chroma_collection and emb_func:
         success = process_and_embed(
             uploaded_files,
             st.session_state.chroma_collection,
             emb_func
         )
         if success:
             st.session_state.processing_status = "success"
             log_message("File processing completed successfully.", "info")
         else:
             st.session_state.processing_status = "error"
             log_message("File processing finished with errors.", "error")
    else:
         st.session_state.processing_status = "error"
         log_message("Processing aborted: Collection or embedding function unavailable after setup.", "error")

    # Optionally clear the file uploader state after processing
    # This is tricky, usually requires rerunning the script.
    # We might need st.rerun() here, but use cautiously in callbacks.


# --- Load or Get Collection Object ---
# Try to load the collection object if it's not already set (e.g., after page refresh)
if st.session_state.chroma_collection is None and st.session_state.collection_name:
    try:
        st.session_state.chroma_collection = chroma_client.get_collection(
            name=st.session_state.collection_name,
            embedding_function=embedding_function
        )
        if "api_configured" in st.session_state: # Only log if API was configured
             log_message(f"Re-attached to existing collection: '{st.session_state.collection_name}'", "info")
    except Exception as e:
        if "api_configured" in st.session_state:
             log_message(f"Collection '{st.session_state.collection_name}' not found yet. Upload files to create it.", "debug")
        st.session_state.chroma_collection = None

# --- Sidebar UI (Now for History/Logs) ---
with st.sidebar:
    st.header("📜 Processing Log")
    # Display processing status indicator
    status = st.session_state.get("processing_status", "idle")
    if status == "processing":
        st.info("⏳ Processing uploaded files...")
    elif status == "success":
        st.success("✅ Processing complete.")
    elif status == "error":
        st.error("⚠️ Processing finished with errors.")

    # Log display area
    log_container = st.container(height=400)
    log_placeholder = log_container.empty()
    log_placeholder.text_area("Logs", "\n".join(st.session_state.processing_log[::-1]), height=400, key="log_display_sidebar", disabled=True)

    st.header("💬 Chat History")
    # Simple display of past user/assistant messages - could be more elaborate
    history_container = st.container(height=300)
    with history_container:
         for msg in st.session_state.messages:
              role = msg.get("role","unknown")
              content_preview = msg.get("content", "")[:50] + "..." # Short preview
              history_container.caption(f"{role.title()}: {content_preview}")


# --- Main Chat Area UI ---
# Display chat messages from history
message_container = st.container()
with message_container:
    if not st.session_state.messages:
        # Initial greeting if no messages yet
        st.chat_message("assistant").write("Hi! Upload some documents below and ask me questions about them.")

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("metadata"):
                 with st.expander("Sources Used", expanded=False):
                      sources = set(meta.get('source', 'Unknown') for meta in message["metadata"] if meta)
                      st.caption("\n".join(f"- {s}" for s in sorted(list(sources))))
            if message["role"] == "assistant" and message.get("follow_ups"):
                 st.markdown("**Suggested Questions:**")
                 cols = st.columns(len(message["follow_ups"]))
                 for j, q in enumerate(message["follow_ups"]):
                      if cols[j].button(q, key=f"followup_{i}_{j}"):
                           st.session_state.rerun_query = q
                           st.rerun()

# --- Input Area ---
input_container = st.container()
with input_container:
    # Handle potential rerun with a follow-up question
    query_input_value = ""
    if st.session_state.rerun_query:
        query_input_value = st.session_state.rerun_query
        st.session_state.rerun_query = None # Clear after use

    # Chat input field
    prompt = st.chat_input(
        "Ask a question about the documents...",
        key="main_query_input",
        disabled=(st.session_state.processing_status == "processing"), # Disable input during processing
        # value=query_input_value # Setting value directly can be buggy with reruns
    )

    # File uploader positioned below the chat input
    uploaded_files = st.file_uploader(
            "Attach Files (Processing starts automatically)",
            accept_multiple_files=True,
            type=ALL_SUPPORTED_TYPES,
            key="doc_uploader_key", # Use a key to access state in callback
            on_change=handle_file_upload, # Set the callback function
            label_visibility="collapsed" # Hide label for cleaner look
        )

    # --- Query Handling ---
    if prompt or query_input_value:
        actual_prompt = prompt or query_input_value

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": actual_prompt})
        # Display immediately using rerun trick or JS? Simpler: let it display on next natural rerun.
        # To display immediately, add to container:
        with message_container:
            with st.chat_message("user"):
                st.markdown(actual_prompt)

        # Check if collection is ready
        if st.session_state.chroma_collection is None:
            st.warning("Hold on! Please wait for files to finish processing or upload files first.")
            log_message("Query attempted before collection was ready.", "warning")
        elif st.session_state.processing_status == "processing":
             st.warning("Hold on! Files are still processing. Please wait.")
        else:
            # Process the query
            with st.spinner("Thinking..."):
                answer = "Sorry, something went wrong processing your query."
                follow_ups = []
                used_metadata = []

                # 1. Retrieve context
                context_docs, context_metadatas = get_relevant_context(
                    actual_prompt,
                    st.session_state.chroma_collection
                )

                # 2. Determine query type and generate answer
                if is_data_query(actual_prompt):
                    log_message("Query classified as data-related.", "info")
                    data_answer = answer_data_query(actual_prompt, context_metadatas)
                    if data_answer:
                        answer = data_answer
                        used_metadata = context_metadatas # Indicate context was used to find source
                    else:
                        log_message("Data query failed or returned no answer, falling back to general context.", "info")
                        answer, used_metadata = generate_answer_from_context(actual_prompt, context_docs, context_metadatas)
                else:
                    log_message("Query classified as general text-based.", "info")
                    answer, used_metadata = generate_answer_from_context(actual_prompt, context_docs, context_metadatas)

                # 3. Generate follow-ups
                if answer and "cannot answer" not in answer.lower() and "error" not in answer.lower() and "blocked" not in answer.lower():
                    follow_ups = generate_followup_questions(actual_prompt, answer)

            # Add assistant response to chat history
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "follow_ups": follow_ups,
                "metadata": used_metadata
            }
            st.session_state.messages.append(assistant_message)

            # Rerun the script to update the chat display AND the log
            st.rerun()


# Update log display one last time at the end of the script run
log_placeholder.text_area("Logs", "\n".join(st.session_state.processing_log[::-1]), height=400, key="log_display_sidebar_final", disabled=True)
