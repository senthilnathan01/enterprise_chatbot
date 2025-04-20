# vector_store.py
"""Handles interaction with ChromaDB: embedding, adding data, and querying."""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import time

# Import necessary functions from other modules
from utils import log_message, chunk_text, generate_unique_id # generate_unique_id now uses UUID
from file_parsers import parse_pdf, parse_docx, parse_pptx, parse_data_file, parse_txt, parse_image
from web_crawler import crawl_url
from config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_CONTEXT_RESULTS,
    VECTOR_DB_PERSIST_PATH,
    ALL_SUPPORTED_TYPES
)

# --- Embedding Function ---
@st.cache_resource
def initialize_embedding_function(api_key):
     """Creates the embedding function object, cached for the session."""
     if "embedding_func_initialized" not in st.session_state:
          log_message("Initializing embedding function...", "debug")
          if not api_key:
               log_message("API Key is missing, cannot initialize embedding function.", "error")
               return None
          try:
               func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                   api_key=api_key,
                   model_name=EMBEDDING_MODEL_NAME
               )
               st.session_state.embedding_func_initialized = True
               log_message(f"Embedding function initialized with model: {EMBEDDING_MODEL_NAME}", "info")
               return func
          except Exception as e:
               log_message(f"Failed to create embedding function: {e}", "error")
               st.exception(e)
               return None

# --- Core Processing and Embedding Pipeline ---
def process_and_embed(uploaded_files, collection, embedding_function):
    """
    Processes uploaded files, extracts content, chunks, embeds, and stores in ChromaDB.
    Uses UUIDs for chunk IDs to prevent duplicates.
    """
    if not uploaded_files:
        log_message("No files provided for processing.", "warning")
        return False
    if collection is None:
         log_message("Cannot process: ChromaDB collection is not available.", "error")
         return False
    if embedding_function is None:
         log_message("Cannot process: Embedding function is not available.", "error")
         return False

    all_chunks_to_add = [] # List of tuples: (id, text_chunk, metadata)
    files_processed_this_run = set()
    total_files = len(uploaded_files)
    processed_count = 0

    if "processed_files" not in st.session_state: st.session_state.processed_files = {}
    if "data_frames" not in st.session_state: st.session_state.data_frames = {}
    if "crawled_data" not in st.session_state: st.session_state.crawled_data = {}

    overall_start_time = time.time()
    log_message(f"--- Starting processing job for {total_files} files ---")

    parser_map = {
        'pdf': parse_pdf, 'docx': parse_docx, 'pptx': parse_pptx,
        'csv': parse_data_file, 'xlsx': parse_data_file, 'json': parse_data_file,
        'txt': parse_txt, 'png': parse_image, 'jpg': parse_image, 'jpeg': parse_image,
    }

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        file_start_time = time.time()
        # Note: Can't use st.info reliably inside callback for progress, rely on logs
        log_message(f"Starting file {i+1}/{total_files}: {filename}")

        # Get current chat state to update processed_files status correctly
        # This assumes process_and_embed is called from a context where current_chat_state is accessible
        # If called independently, this needs adjustment. Let's assume it's called from main_app's callback.
        current_chat_state = st.session_state.chats.get(st.session_state.current_chat_id)
        if not current_chat_state:
            log_message(f"Could not get current chat state while processing {filename}. Aborting file.", "error")
            continue # Skip this file if chat state is invalid

        if current_chat_state["processed_files"].get(filename) == 'success':
             log_message(f"Skipping already successfully processed file in this chat: {filename}", "info")
             continue

        file_content = uploaded_file.getvalue()
        file_type = filename.split('.')[-1].lower()
        parser_func = parser_map.get(file_type)

        if not parser_func:
            log_message(f"Unsupported file type: {filename}. Skipping.", "warning")
            current_chat_state["processed_files"][filename] = 'skipped'
            continue

        # --- File Parsing ---
        try:
            parsed_segments, urls = parser_func(file_content, filename)

            if not parsed_segments:
                if file_type in ['csv', 'xlsx', 'json'] and filename in current_chat_state.get("data_frames", {}):
                     log_message(f"No text summary for data file '{filename}', but data object exists.", "info")
                else:
                     log_message(f"No content segments extracted from {filename}. Marking as failed.", "warning")
                     current_chat_state["processed_files"][filename] = 'failed'
                     continue

            # --- Crawl URLs ---
            crawled_segments = []
            if urls:
                unique_urls = set(u for u in urls if u)
                log_message(f"Found {len(unique_urls)} unique URLs in {filename}. Attempting crawl...", "info")
                for url in unique_urls:
                    crawled_text = crawl_url(url)
                    if crawled_text:
                         crawl_meta = {"source": filename, "crawled_url": url, "content_type": "crawled_web"}
                         crawled_segments.append((f"\n--- Content from {url} ---\n{crawled_text}\n--- End Content ---", crawl_meta))

            # --- Combine Parsed and Crawled Segments ---
            all_segments_for_file = parsed_segments + crawled_segments

            # --- Chunking Each Segment ---
            for text_segment, segment_meta in all_segments_for_file:
                 if not text_segment or not isinstance(text_segment, str): continue
                 chunks = chunk_text(text_segment, CHUNK_SIZE, CHUNK_OVERLAP)

                 for chunk_index, chunk in enumerate(chunks):
                      # *** USE UUID FOR ID ***
                      chunk_id = generate_unique_id() # Generates a unique UUID string
                      chunk_meta = segment_meta.copy()
                      chunk_meta["chunk_index"] = chunk_index
                      chunk_meta["segment_length"] = len(text_segment)
                      chunk_meta["chunk_in_segment"] = f"{chunk_index + 1}/{len(chunks)}"
                      # Ensure basic metadata always exists
                      chunk_meta.setdefault("source", filename)
                      chunk_meta.setdefault("content_type", "unknown")

                      all_chunks_to_add.append((chunk_id, chunk, chunk_meta))

            files_processed_this_run.add(filename)
            current_chat_state["processed_files"][filename] = 'success' # Mark success for this file
            file_end_time = time.time()
            log_message(f"Finished file: {filename} (took {file_end_time - file_start_time:.2f}s)", "info")

        except Exception as e:
            st.exception(e)
            log_message(f"Critical error during processing pipeline for {filename}: {e}", "error")
            current_chat_state["processed_files"][filename] = 'failed'

        processed_count += 1

    # --- Batch Add to ChromaDB ---
    if all_chunks_to_add:
        ids_batch = [item[0] for item in all_chunks_to_add]
        docs_batch = [item[1] for item in all_chunks_to_add]
        metadatas_batch = [item[2] for item in all_chunks_to_add]

        # Check for duplicates within the batch before sending to ChromaDB (optional sanity check)
        if len(ids_batch) != len(set(ids_batch)):
             log_message("Error: Duplicate IDs generated within the same batch before adding to ChromaDB!", "error")
             # Find duplicates for logging
             from collections import Counter
             id_counts = Counter(ids_batch)
             duplicates = {id: count for id, count in id_counts.items() if count > 1}
             log_message(f"Duplicate IDs found: {duplicates}", "error")
             return False # Fail processing if internal duplication occurs

        batch_size = 100
        num_batches = (len(ids_batch) + batch_size - 1) // batch_size
        log_message(f"Preparing to add {len(ids_batch)} chunks to vector store in {num_batches} batches...")
        add_errors = 0
        add_start_time = time.time()

        try:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(ids_batch))
                batch_ids = ids_batch[start_idx:end_idx]
                batch_docs = docs_batch[start_idx:end_idx]
                batch_metas = metadatas_batch[start_idx:end_idx]

                log_message(f"Adding batch {i+1}/{num_batches} ({len(batch_ids)} chunks)...", "debug")
                # Use upsert=True maybe? Or handle potential existing IDs if needed?
                # For a fresh collection per chat, 'add' should be fine.
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

            add_end_time = time.time()
            log_message(f"Finished adding {len(ids_batch)} chunks to vector store (took {add_end_time - add_start_time:.2f}s).", "info")

        except Exception as e:
            add_errors += 1
            st.exception(e)
            log_message(f"Error adding batch {i+1 if 'i' in locals() else '?'} to ChromaDB: {e}", "error")
            # If one batch fails, should we mark the whole process as failed? Yes.
            return False # Indicate failure

        if add_errors > 0:
             log_message(f"Completed adding chunks with {add_errors} batch errors.", "error")
             return False
        else:
             log_message("All batches added successfully.", "info")
             return True

    else:
        log_message("No new chunks were generated to add to the vector store.", "info")
        # Check if any file succeeded even without chunks (e.g., data summary only)
        return any(status == 'success' for filename, status in current_chat_state["processed_files"].items() if filename in files_processed_this_run)


def get_relevant_context(query, collection, n_results=MAX_CONTEXT_RESULTS):
    """Retrieves relevant text chunks from ChromaDB."""
    if collection is None:
        log_message("Cannot retrieve context, collection is None.", "error")
        return [], []
    try:
        log_message(f"Querying vector store for '{query[:50]}...' (n_results={n_results})", "debug")
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        if results and results.get('ids') and results['ids'][0]:
             combined = list(zip(
                 results.get('documents', [[]])[0],
                 results.get('metadatas', [[]])[0],
                 results.get('distances', [[]])[0]
             ))
             valid_results = [(doc, meta, dist) for doc, meta, dist in combined if dist is not None]
             sorted_results = sorted(valid_results, key=lambda x: x[2])

             context_docs = [item[0] for item in sorted_results]
             context_metadatas = [item[1] for item in sorted_results]
             log_message(f"Retrieved {len(context_docs)} relevant context chunks.", "debug")
             return context_docs, context_metadatas
        else:
            log_message("No relevant context found in vector store for the query.", "info")
            return [], []
    except Exception as e:
        st.exception(e)
        log_message(f"Error querying ChromaDB: {e}", "error")
        return [], []

