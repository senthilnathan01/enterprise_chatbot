# web_crawler.py
"""Handles fetching and parsing content from web URLs."""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from utils import log_message, find_urls # find_urls might be only needed by parsers
from config import CRAWLING_TIMEOUT, MAX_CRAWL_TEXT_LENGTH

def crawl_url(url):
    """
    Fetches and parses text content from a URL.
    Uses session state for caching results.
    """
    if "crawled_data" not in st.session_state:
        st.session_state.crawled_data = {}

    # Normalize URL slightly? (e.g., remove fragment)
    normalized_url = url.split('#')[0]

    if normalized_url in st.session_state.crawled_data:
        log_message(f"Using cached content for URL: {normalized_url}", "debug") # Use debug level
        return st.session_state.crawled_data[normalized_url] # Return cached data (could be None if failed before)

    log_message(f"Crawling URL: {normalized_url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; QABot/1.0; +http://example.com/bot)'} # Be a good bot citizen
        response = requests.get(normalized_url, timeout=CRAWLING_TIMEOUT, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        content_type = response.headers.get('content-type', '').lower()

        # Handle redirects? requests usually does this automatically. Check response.url vs normalized_url

        if 'html' in content_type:
            soup = BeautifulSoup(response.content, 'lxml') # Use lxml parser

            # Remove common clutter elements before extracting text
            for element_type in ["script", "style", "nav", "footer", "aside", "header", "form", "button", "input"]:
                 for element in soup.find_all(element_type):
                      element.decompose()

            # Get text from main content areas if possible (requires site-specific logic, difficult)
            # Generic approach: extract all stripped strings
            text = ' '.join(soup.stripped_strings)
            text = text[:MAX_CRAWL_TEXT_LENGTH] # Limit length
            st.session_state.crawled_data[normalized_url] = text # Cache result
            log_message(f"Successfully crawled and parsed {len(text)} chars from {normalized_url}")
            return text
        elif 'text/plain' in content_type:
             text = response.text[:MAX_CRAWL_TEXT_LENGTH]
             st.session_state.crawled_data[normalized_url] = text
             log_message(f"Successfully crawled plain text from {normalized_url}")
             return text
        else:
            log_message(f"Skipping non-HTML/text content at {normalized_url} (type: {content_type})", "warning")
            st.session_state.crawled_data[normalized_url] = None # Cache failure
            return None
    except requests.exceptions.Timeout:
        log_message(f"Failed to crawl {normalized_url}: Timeout after {CRAWLING_TIMEOUT} seconds", "error")
        st.session_state.crawled_data[normalized_url] = None
        return None
    except requests.exceptions.RequestException as e:
        log_message(f"Failed to crawl {normalized_url}: {e}", "error")
        st.session_state.crawled_data[normalized_url] = None
        return None
    except Exception as e:
        log_message(f"Error parsing content from {normalized_url}: {e}", "error")
        st.session_state.crawled_data[normalized_url] = None
        return None