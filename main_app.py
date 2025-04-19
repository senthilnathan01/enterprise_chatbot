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
    page_title="Modular Multi-Modal Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Modular Multi-Modal Q&A Chatbot")
st.caption("Upload Documents & Ask Questions (Supports Text, Data, Images, Links, OCR)")

# --- API Key and Model Configuration ---
# Get API Key (Prefer environment variables/secrets)
try:
    google_api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.warning("Google API Key not found in environment variables or Streamlit secrets.", icon="⚠️")
    google_api_key = st.text_input("Enter your Google AI API Key:", type="password", key="api_key_input_main")

if not google_api_key:
    st.info("Please provide your Google AI API Key in the sidebar or environment to proceed.")
    st.stop()

# Configure Google AI API
try:
    genai.configure(api_key=google_api_key)
    log_message("Google AI API configured successfully.", "info")
except Exception as e:
    st.error(f"Fatal Error: Could not configure Google AI API: {e}")
    log_message(f"Fatal Error: Could not configure Google AI API: {e}", "error")
    st.stop()

# --- Initialize ChromaDB Client and Embedding Function ---
# Cache the client and embedding function for the session
@st.cache_resource
def initialize_chroma_client():
    log_message("Initializing ChromaDB client...", "debug")
    # Use persistence if path is configured
    if VECTOR_DB_PERSIST_PATH:
         log_message(f"Using persistent ChromaDB storage at: {VECTOR_DB_PERSIST_PATH}", "info")
         return chromadb.PersistentClient(path=VECTOR_DB_PERSIST_PATH)
    else:
         log_message("Using in-memory ChromaDB storage.", "info")
         return chromadb.Client()

@st.cache_resource
def initialize_embedding_function(api_key):
     log_message("Initializing embedding function...", "debug")
     return get_embedding_function(api_key)

chroma_client = initialize_chroma_client()
embedding_function = initialize_embedding_function(google_api_key)

if embedding_function is None:
     st.error("Fatal Error: Could not initialize embedding function. Check API Key and model access.")
     st.stop()


# --- Session State Management ---
def initialize_session_state():
    """Initializes required session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        log_message(f"New session started: {st.session_state.session_id}", "info")
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = DEFAULT_COLLECTION_NAME_PREFIX + st.session_state.session_id
    if "chroma_collection" not in st.session_state: st.session_state.chroma_collection = None
    if "processed_files" not in st.session_state: st.session_state.processed_files = {} # {filename: status}
    if "messages" not in st.session_state: st.session_state.messages = [] # Chat history
    if "processing_log" not in st.session_state: st.session_state.processing_log = []
    if "data_frames" not in st.session_state: st.session_state.data_frames = {} # Stores parsed pandas DFs or JSON data
    if "crawled_data" not in st.session_state: st.session_state.crawled_data = {} # Caches crawled URL content
    if "rerun_query" not in st.session_state: st.session_state.rerun_query = None # For follow-up clicks

initialize_session_state()

# Ensure collection object is loaded if name exists in state (handles page reloads)
if st.session_state.chroma_collection is None and st.session_state.collection_name:
    try:
        # Try to get the existing collection for this session
        st.session_state.chroma_collection = chroma_client.get_collection(
            name=st.session_state.collection_name,
            embedding_function=embedding_function # Must match original
        )
        log_message(f"Re-attached to existing collection: '{st.session_state.collection_name}'", "info")
    except Exception as e:
        # Collection might not exist yet or other issue
        log_message(f"Could not get collection '{st.session_state.collection_name}'. Will create on process. Error: {e}", "debug")
        st.session_state.chroma_collection = None


# --- Sidebar UI ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Files",
        accept_multiple_files=True,
        type=ALL_SUPPORTED_TYPES,
        key="file_uploader"
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected. Click below to process.")
        # Add a button to clear existing processed data before processing new files
        if st.button("Clear Existing & Process Files", key="process_button"):
            with st.spinner("Clearing old data and processing new files..."):
                # --- Reset relevant session state ---
                st.session_state.messages = [] # Clear chat
                st.session_state.processed_files = {}
                st.session_state.data_frames = {}
                st.session_state.crawled_data = {}
                st.session_state.processing_log = [] # Clear log for new job

                # --- Delete and recreate Chroma collection ---
                collection_name = st.session_state.collection_name
                try:
                    log_message(f"Attempting to delete collection: '{collection_name}'.", "info")
                    chroma_client.delete_collection(name=collection_name)
                    log_message(f"Collection '{collection_name}' deleted.", "info")
                except Exception as e:
                    log_message(f"Collection '{collection_name}' not found or could not be deleted (might be first run): {e}", "warning")

                # Create the new collection
                try:
                     st.session_state.chroma_collection = chroma_client.get_or_create_collection(
                         name=collection_name,
                         embedding_function=embedding_function,
                         metadata={"hnsw:space": "cosine"} # Example metadata
                     )
                     log_message(f"Created/recreated collection: '{collection_name}'.", "info")
                except Exception as e:
                     st.error(f"Fatal Error: Failed to create ChromaDB collection: {e}")
                     log_message(f"Fatal Error: Failed to create ChromaDB collection: {e}", "error")
                     st.stop() # Stop execution if collection fails


                # --- Process Files ---
                if st.session_state.chroma_collection and embedding_function:
                     success = process_and_embed(
                         uploaded_files,
                         st.session_state.chroma_collection,
                         embedding_function
                     )
                     if success:
                         st.success("✅ Files processed successfully!")
                     else:
                         st.error("⚠️ File processing finished with errors. Check logs.")
                else:
                     st.error("Processing aborted: Collection or embedding function not available.")

            # Clear the uploader widget after processing
            # st.experimental_rerun() # Might be needed if uploader state doesn't clear nicely


    st.header("📊 Processing Status")
    # Display summary of processed files
    if st.session_state.processed_files:
         successful_files = [f for f, status in st.session_state.processed_files.items() if status == 'success']
         failed_files = [f for f, status in st.session_state.processed_files.items() if status == 'failed']
         skipped_files = [f for f, status in st.session_state.processed_files.items() if status == 'skipped']
         st.caption(f"✓ {len(successful_files)} succeeded, ! {len(failed_files)} failed, - {len(skipped_files)} skipped/unsupported.")
         with st.expander("Show Processed File Details"):
              st.json(st.session_state.processed_files)
    else:
         st.caption("No files processed in this session yet.")


    st.header("📜 Processing Log")
    log_container = st.container(height=300)
    log_placeholder = log_container.empty()
    log_placeholder.text_area("Logs", "\n".join(st.session_state.processing_log[::-1]), height=300, key="log_display_sidebar", disabled=True)


# --- Main Chat Area UI ---
st.header("💬 Ask Questions")

# Display chat history
message_container = st.container() # Use a container to manage message display
with message_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources for assistant messages
            if message["role"] == "assistant" and message.get("metadata"):
                 with st.expander("Sources Used", expanded=False):
                      sources = set(meta.get('source', 'Unknown') for meta in message["metadata"] if meta) # Handle potential None metadata
                      st.caption("\n".join(f"- {s}" for s in sorted(list(sources))))
            # Display follow-up buttons
            if message["role"] == "assistant" and message.get("follow_ups"):
                 st.markdown("**Suggested Questions:**")
                 cols = st.columns(len(message["follow_ups"]))
                 for j, q in enumerate(message["follow_ups"]):
                      if cols[j].button(q, key=f"followup_{i}_{j}"):
                           st.session_state.rerun_query = q
                           st.rerun() # Trigger rerun with the selected question

# Handle potential rerun with a follow-up question
query_input_value = ""
if st.session_state.rerun_query:
    query_input_value = st.session_state.rerun_query
    st.session_state.rerun_query = None # Clear after use


# Chat input
if prompt := st.chat_input("Ask a question about the documents...", key="main_query_input", disabled=not st.session_state.processed_files) or query_input_value:

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with message_container: # Display immediately in the container
         with st.chat_message("user"):
              st.markdown(prompt)

    # Check if collection is ready
    if st.session_state.chroma_collection is None:
        st.warning("Vector store collection not initialized. Please process files first.")
        log_message("Query attempted before collection was ready.", "warning")
    else:
        # Process the query
        with st.spinner("Thinking..."):
            answer = "Sorry, something went wrong." # Default answer
            follow_ups = []
            used_metadata = []

            # 1. Retrieve relevant context
            context_docs, context_metadatas = get_relevant_context(
                prompt,
                st.session_state.chroma_collection
            )

            # 2. Determine query type and generate answer
            if is_data_query(prompt):
                 log_message("Query classified as data-related.", "info")
                 data_answer = answer_data_query(prompt, context_metadatas)
                 if data_answer:
                      answer = data_answer
                      used_metadata = context_metadatas # Metadata primarily used to find the source
                 else:
                      # Fallback to general context if data query fails
                      log_message("Data query failed or returned no answer, falling back to general context.", "info")
                      answer, used_metadata = generate_answer_from_context(prompt, context_docs, context_metadatas)
            else:
                 # General query using RAG
                 log_message("Query classified as general text-based.", "info")
                 answer, used_metadata = generate_answer_from_context(prompt, context_docs, context_metadatas)

            # 3. Generate follow-up questions (only if answer was successful)
            if answer and "cannot answer" not in answer.lower() and "error" not in answer.lower() and "blocked" not in answer.lower():
                follow_ups = generate_followup_questions(prompt, answer)


        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "follow_ups": follow_ups,
            "metadata": used_metadata # Store metadata used for this answer
        }
        st.session_state.messages.append(assistant_message)

        # Display assistant response in the container
        with message_container:
            with st.chat_message("assistant"):
                st.markdown(assistant_message["content"])
                if assistant_message.get("metadata"):
                     with st.expander("Sources Used", expanded=False):
                          sources = set(meta.get('source', 'Unknown') for meta in assistant_message["metadata"] if meta)
                          st.caption("\n".join(f"- {s}" for s in sorted(list(sources))))
                if assistant_message.get("follow_ups"):
                     st.markdown("**Suggested Questions:**")
                     cols = st.columns(len(assistant_message["follow_ups"]))
                     # Use current message index and suggestion index for keys
                     current_msg_index = len(st.session_state.messages) - 1 # Index of the message just added
                     for i, q in enumerate(assistant_message["follow_ups"]):
                          if cols[i].button(q, key=f"followup_{current_msg_index}_{i}"):
                               st.session_state.rerun_query = q
                               st.rerun()

    # Update the log display after processing query
    log_placeholder.text_area("Logs", "\n".join(st.session_state.processing_log[::-1]), height=300, key="log_display_sidebar_update", disabled=True)


# Add disclaimer or instructions at the bottom
st.divider()
st.caption("AI responses are based solely on the content of processed documents and may require interpretation. Always verify critical information.")