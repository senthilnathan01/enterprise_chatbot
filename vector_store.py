# vector_store.py
"""Handles interaction with ChromaDB: embedding, adding data, and querying."""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import time

# Import necessary functions from other modules
from utils import log_message, chunk_text, generate_unique_id
from file_parsers import parse_pdf, parse_docx, parse_pptx, parse_data_file, parse_txt, parse_image
from web_crawler import crawl_url
from config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_CONTEXT_RESULTS,
    VECTOR_DB_PERSIST_PATH # Added for optional persistence
)

# Note: Assumes google_api_key is available via st.session_state or config
# Note: Assumes ChromaDB client is initialized in main_app


def get_embedding_function(api_key):
     """Creates the embedding function object."""
     if not api_key:
          log_message("API Key needed for embedding function.", "error")
          return None
     try:
          # Consider making this a cached resource too if init is slow
          return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
              api_key=api_key,
              model_name=EMBEDDING_MODEL_NAME
          )
     except Exception as e:
          log_message(f"Failed to create embedding function: {e}", "error")
          return None


def process_and_embed(uploaded_files, collection, embedding_function):
    """
    Processes uploaded files, extracts content, chunks, embeds, and stores in ChromaDB.
    Takes collection and embedding_function as arguments.
    """
    if not uploaded_files:
        log_message("No files provided for processing.", "warning")
        return False
    if collection is None:
         log_message("ChromaDB collection not available.", "error")
         return False
    if embedding_function is None:
         log_message("Embedding function not available.", "error")
         return False


    all_chunks_to_add = [] # List of tuples: (id, text_chunk, metadata)
    files_processed_this_run = set()
    total_files = len(uploaded_files)
    processed_count = 0

    # Initialize session state variables if they don't exist
    if "processed_files" not in st.session_state: st.session_state.processed_files = {}
    if "data_frames" not in st.session_state: st.session_state.data_frames = {}
    if "crawled_data" not in st.session_state: st.session_state.crawled_data = {}


    with st.spinner(f"Processing {total_files} files... This may take time."):
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            st.info(f"Processing file {i+1}/{total_files}: {filename}...") # Give progress feedback

            if st.session_state.processed_files.get(filename) == 'success':
                 log_message(f"Skipping already successfully processed file: {filename}", "info")
                 continue

            log_message(f"--- Starting processing for: {filename} ---")
            file_content = uploaded_file.getvalue()
            text = ""
            images_data = [] # Text analysis/OCR from images within file
            urls = []
            file_type = filename.split('.')[-1].lower()
            data_obj = None # For storing parsed structured data (DF or dict)

            # --- File Parsing ---
            try:
                if file_type == "pdf":
                    text, images_data, urls = parse_pdf(file_content, filename)
                elif file_type == "docx":
                    text, images_data, urls = parse_docx(file_content, filename)
                elif file_type == "pptx":
                     text, images_data, urls = parse_pptx(file_content, filename)
                elif file_type in ["csv", "xlsx", "json"]:
                    # parse_data_file returns summary text and the data object
                    text, data_obj = parse_data_file(file_content, filename)
                    # Store data object in session state (done within the parser now)
                elif file_type == "txt":
                    text, _, urls = parse_txt(file_content, filename)
                elif file_type in ["png", "jpg", "jpeg"]:
                    text, images_data, _ = parse_image(file_content, filename)
                else:
                    log_message(f"Unsupported file type: {filename}. Skipping.", "warning")
                    st.session_state.processed_files[filename] = 'skipped'
                    continue

                # Check if parsing failed (returned empty text and no data object)
                if not text and data_obj is None:
                    log_message(f"No text or data extracted from {filename}. Marking as failed.", "warning")
                    st.session_state.processed_files[filename] = 'failed'
                    continue

                # --- Crawl URLs ---
                crawled_texts = []
                if urls:
                    unique_urls = set(u.split('#')[0] for u in urls if u) # Normalize and deduplicate
                    log_message(f"Found {len(unique_urls)} unique URLs in {filename}. Attempting crawl...", "info")
                    crawl_progress = st.progress(0)
                    for idx, url in enumerate(unique_urls):
                         # Avoid re-crawling excessively within the same run if URL appears in multiple docs
                         if url not in st.session_state.crawled_data or st.session_state.crawled_data[url] is None:
                              crawled_content = crawl_url(url) # Caches result in session state
                         else:
                              crawled_content = st.session_state.crawled_data[url] # Use cache

                         if crawled_content:
                              crawled_texts.append(f"\n--- Content from {url} ---\n{crawled_content}\n--- End Content from {url} ---\n")
                         crawl_progress.progress((idx + 1) / len(unique_urls))
                    crawl_progress.empty() # Remove progress bar

                # --- Combine Text & Chunk ---
                # Include image analysis text with the main text
                image_analysis_text = "\n\n".join(images_data)
                full_text_for_embedding = text + "\n\n" + image_analysis_text + "\n\n" + "\n\n".join(crawled_texts)

                chunks = chunk_text(full_text_for_embedding, CHUNK_SIZE, CHUNK_OVERLAP)
                log_message(f"Split content for '{filename}' (text, images, crawled) into {len(chunks)} chunks.", "info")

                if not chunks:
                     # If there's structured data, we might still proceed if the summary chunk exists
                     if data_obj is None:
                          log_message(f"No content chunks generated for {filename}. Marking as failed.", "warning")
                          st.session_state.processed_files[filename] = 'failed'
                          continue
                     else:
                           log_message(f"No text chunks generated for {filename}, but data summary exists.", "info")


                # --- Prepare Chunks for Batch ---
                # Add text/image/crawled chunks
                for i, chunk in enumerate(chunks):
                    chunk_id = generate_unique_id(f"{filename}_chunk{i}")
                    metadata = {"source": filename, "chunk_index": i, "type": "content"}
                    all_chunks_to_add.append((chunk_id, chunk, metadata))

                # Add data summary as a chunk if data was parsed
                if data_obj is not None and text: # 'text' holds the summary here
                    data_summary_id = generate_unique_id(f"{filename}_datasummary")
                    data_summary_metadata = {"source": filename, "chunk_index": 0, "type": "data_summary"}
                    all_chunks_to_add.append((data_summary_id, text, data_summary_metadata))

                files_processed_this_run.add(filename)
                st.session_state.processed_files[filename] = 'success' # Mark success for this file
                log_message(f"--- Finished processing for: {filename} ---", "info")


            except Exception as e:
                st.exception(e) # Show full traceback in Streamlit for debugging
                log_message(f"Critical error during processing pipeline for {filename}: {e}", "error")
                st.session_state.processed_files[filename] = 'failed' # Mark as failed

            processed_count += 1
            # Update overall progress if needed, though spinner shows activity

    # --- Batch Add to ChromaDB ---
    if all_chunks_to_add:
        ids_batch = [item[0] for item in all_chunks_to_add]
        docs_batch = [item[1] for item in all_chunks_to_add]
        metadatas_batch = [item[2] for item in all_chunks_to_add]

        # Batching for ChromaDB (add in smaller groups if very large)
        batch_size = 100 # Adjust as needed based on performance/API limits
        num_batches = (len(ids_batch) + batch_size - 1) // batch_size
        log_message(f"Preparing to add {len(ids_batch)} chunks to vector store in {num_batches} batches...")

        with st.spinner(f"Adding {len(ids_batch)} content chunks to vector database..."):
             try:
                  for i in range(num_batches):
                       start_idx = i * batch_size
                       end_idx = min((i + 1) * batch_size, len(ids_batch))
                       log_message(f"Adding batch {i+1}/{num_batches} ({end_idx - start_idx} chunks)...", "debug")

                       collection.add(
                           ids=ids_batch[start_idx:end_idx],
                           documents=docs_batch[start_idx:end_idx],
                           metadatas=metadatas_batch[start_idx:end_idx]
                       )
                       # Optional: small delay between batches if hitting API rate limits
                       # time.sleep(0.5)

                  log_message(f"Successfully added {len(ids_batch)} chunks to the vector store.", "info")
                  return True # Indicate success
             except Exception as e:
                  st.exception(e)
                  log_message(f"Error adding documents to ChromaDB: {e}", "error")
                  # Should we mark files as failed if embedding fails? Difficult decision.
                  return False # Indicate failure
    else:
        log_message("No new valid chunks were generated to add to the vector store.", "info")
        # If no chunks were added, but some files might have been processed (e.g., only data files)
        # return True if len(files_processed_this_run) > 0 else False
        return True # Considered success if processing ran without critical errors, even if no new chunks


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
            include=['documents', 'metadatas', 'distances'] # Request necessary fields
        )

        # Process results safely
        if results and results.get('ids') and results['ids'][0]: # Check if non-empty results
             # Combine results into tuples: (doc, meta, distance)
             combined = list(zip(
                 results.get('documents', [[]])[0],
                 results.get('metadatas', [[]])[0],
                 results.get('distances', [[]])[0]
             ))
             # Sort by distance (ascending)
             sorted_results = sorted(combined, key=lambda x: x[2] if x[2] is not None else float('inf'))

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