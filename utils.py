# utils.py
"""General utility functions."""

import streamlit as st
import time
import random
import string
import re

def log_message(message, level="info"):
    """Adds message to the session's processing log."""
    if "processing_log" not in st.session_state:
        st.session_state.processing_log = []
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}][{level.upper()}] {message}"
    st.session_state.processing_log.append(log_entry)
    # Optionally display errors/warnings immediately in the UI if needed
    # if level == "error": st.error(message)
    # elif level == "warning": st.warning(message)

def generate_unique_id(prefix="doc"):
    """Generates a unique ID using timestamp and random numbers."""
    return f"{prefix}_{int(time.time())}_{random.randint(1000, 9999)}"

def find_urls(text):
    """Finds potential URLs in text using regex."""
    if not isinstance(text, str):
        return []
    # Simple regex, consider refining for edge cases
    try:
        return re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)
    except Exception as e:
        log_message(f"Error finding URLs: {e}", "warning")
        return []

def chunk_text(text, chunk_size, chunk_overlap):
    """Splits text into overlapping chunks."""
    if not isinstance(text, str) or not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        # Basic safeguards
        if start >= len(text): break
        if start < 0: start = end
    return chunks