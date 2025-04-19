# config.py
"""Stores configuration variables and constants."""

# --- Model Names ---
TEXT_MODEL_NAME = "gemini-2.0-flash"
VISION_MODEL_NAME = "gemini-pro-vision"
EMBEDDING_MODEL_NAME = "models/embedding-001" # Or specific embedding model like "text-embedding-004"

# --- Processing Constants ---
CRAWLING_TIMEOUT = 10 # Seconds for web requests
MAX_CRAWL_TEXT_LENGTH = 5000 # Limit crawled text size per URL
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTEXT_RESULTS = 7 # Number of chunks to retrieve for context

# --- ChromaDB Settings ---
# Collection name generated dynamically in main_app.py based on session
DEFAULT_COLLECTION_NAME_PREFIX = "qa_coll_"
VECTOR_DB_PERSIST_PATH = None # Set to a directory path for persistence, None for in-memory

# --- File Type Handling ---
SUPPORTED_TEXT_TYPES = ['pdf', 'docx', 'pptx', 'txt']
SUPPORTED_DATA_TYPES = ['csv', 'xlsx', 'json']
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg']
ALL_SUPPORTED_TYPES = SUPPORTED_TEXT_TYPES + SUPPORTED_DATA_TYPES + SUPPORTED_IMAGE_TYPES