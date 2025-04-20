# qa_engine.py
"""
Handles the core question answering logic, including data queries and follow-ups.
V2 Changes:
- Functions now accept `text_model_name` argument to use the selected LLM.
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import io # Import io for StringIO used in answer_data_query
from utils import log_message
# Removed config import for TEXT_MODEL_NAME as it's now passed in

# Assumes genai is configured in main_app scope

def is_data_query(query):
    """Basic keyword-based check if a query seems related to structured data."""
    data_keywords = [
        "total", "average", "mean", "sum", "count", "list", "show me data",
        "maximum", "minimum", "value of", "in table", "from csv", "in excel",
        "json record", "how many rows", "column names", "what are the columns",
        "calculate", "statistics for", "summarize the data", "plot", "chart", "graph"
    ]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in data_keywords):
        return True
    return False

def answer_data_query(query, relevant_metadatas, text_model_name): # Added text_model_name argument
    """
    Attempts to answer a query using stored DataFrames or JSON data.
    Uses the specified LLM to interpret the question against the data description/summary.
    """
    data_source_files = set()
    primary_sources = set(meta['source'] for meta in relevant_metadatas if meta and meta.get('type') == 'data_summary')
    secondary_sources = set(meta['source'] for meta in relevant_metadatas if meta and meta.get('source') and meta.get('source').split('.')[-1].lower() in ['csv', 'xlsx', 'json'])
    data_source_files.update(primary_sources)
    if not primary_sources:
        data_source_files.update(secondary_sources)

    if not data_source_files:
        log_message("Data query routing: No relevant data source file identified in context.", "debug")
        return None

    data_descriptions = []
    available_data_files = []
    current_chat_state = get_current_chat_state() # Need helper from main_app or pass state

    # --- Need access to current chat state ---
    # This function ideally shouldn't rely on get_current_chat_state directly
    # It's better if main_app passes the relevant data_frames dict
    # For now, let's assume access via session_state (requires helper/modification)
    if not current_chat_state:
         log_message("Cannot answer data query: could not get current chat state.", "error")
         return None
    # --- End dependency issue ---

    for target_file in data_source_files:
         # Access dataframes specific to the current chat
         if target_file in current_chat_state.get("data_frames", {}):
             available_data_files.append(target_file)
             data_content = current_chat_state["data_frames"][target_file]
             description = ""
             try:
                 if isinstance(data_content, pd.DataFrame):
                     description = f"Data from file '{target_file}' (CSV/XLSX - DataFrame):\n"
                     description += f"Shape (rows, columns): {data_content.shape}\n"
                     buffer = io.StringIO()
                     data_content.info(buf=buffer, verbose=False) # Concise info
                     description += f"Columns & Types Overview:\n{buffer.getvalue()}\n"
                     description += f"First 3 Rows Preview:\n{data_content.head(3).to_string()}\n"
                 elif isinstance(data_content, (dict, list)):
                     description = f"Data from file '{target_file}' (JSON - type: {type(data_content)}):\n"
                     preview = json.dumps(data_content, indent=2)[:500] # Shorter preview for prompt
                     description += f"Data Preview:\n{preview}...\n"
                 else:
                      description = f"Data source '{target_file}' contains data of type: {type(data_content)}."
                 data_descriptions.append(description)
             except Exception as desc_err:
                  log_message(f"Error creating description for data query {target_file}: {desc_err}", "warning")
                  data_descriptions.append(f"Could not fully describe data from {target_file}.")
         else:
              log_message(f"Dataframe/JSON for {target_file} mentioned in context but not found in current chat state.", "warning")

    if not data_descriptions:
         log_message("Could not generate descriptions for any relevant data files.", "warning")
         return None

    full_data_context = "\n---\n".join(data_descriptions)
    log_message(f"Attempting data query using data from: {', '.join(available_data_files)}", "info")

    prompt = f"""You are a data analysis assistant... (rest of prompt same as before) ...

    **Data Descriptions:**
    ```
    {full_data_context}
    ```

    **User Question:**
    ```
    {query}
    ```

    **Answer:**
    """

    try:
        # *** Use the passed model name ***
        model = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        response = model.generate_content(prompt, generation_config=generation_config)

        if hasattr(response, 'text'):
            log_message("Generated answer using data query logic.", "info")
            return response.text
        # ... (rest of error/block handling same as before) ...
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             log_message(f"Data query answer blocked: {response.prompt_feedback.block_reason}", "warning")
             return f"Could not answer data query due to safety settings: {response.prompt_feedback.block_reason}"
        else:
             log_message("LLM did not generate an answer for the data query.", "warning")
             return None
    except Exception as e:
        st.exception(e)
        log_message(f"Error generating data query answer with LLM {text_model_name}: {e}", "error")
        return f"An error occurred while trying to answer the data query: {e}"


def generate_answer_from_context(query, context_docs, context_metadatas, text_model_name): # Added text_model_name
    """Generates answer using the specified LLM based purely on retrieved text context (RAG)."""
    if not context_docs:
        log_message("No relevant context found in documents for the query.", "warning")
        return "I cannot answer this question based on the provided documents.", []

    context_items = []
    source_files = set()
    for doc, meta in zip(context_docs, context_metadatas):
         if not meta: continue
         source = meta.get('source', 'Unknown')
         source_files.add(source)
         citation = f"Source: {source}"
         if meta.get('page_number'): citation += f" (Page {meta.get('page_number')})"
         elif meta.get('slide_number'): citation += f" (Slide {meta.get('slide_number')})"
         elif meta.get('crawled_url'): citation += f" (From URL: {meta.get('crawled_url')})"
         context_items.append(f"{citation}\nContent:\n{doc}")
    context_str = "\n\n---\n\n".join(context_items)

    prompt = f"""Answer the following question based *only* on the provided context... (rest of prompt same as before) ...

    **Provided Context:**
    ```
    {context_str}
    ```

    **User Question:**
    ```
    {query}
    ```

    **Answer (with citations):**
    """

    try:
        # *** Use the passed model name ***
        model = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.3)
        response = model.generate_content(prompt, generation_config=generation_config)

        if hasattr(response, 'text'):
            answer_text = response.text
            # ... (source citation logic same as before) ...
            if not any(f in answer_text for f in source_files) and len(source_files) < 4:
                 answer_text += f"\n\n(Sources consulted: {', '.join(sorted(list(source_files)))})"
            return answer_text, context_metadatas
        # ... (rest of error/block handling same as before) ...
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             log_message(f"General answer generation blocked: {response.prompt_feedback.block_reason}", "warning")
             return f"Answer blocked due to safety settings: {response.prompt_feedback.block_reason}", context_metadatas
        else:
             log_message("LLM did not generate a general answer.", "warning")
             return "Sorry, I couldn't generate an answer based on the context.", context_metadatas
    except Exception as e:
        st.exception(e)
        log_message(f"Error generating general answer with LLM {text_model_name}: {e}", "error")
        return f"An error occurred while generating the answer: {e}", context_metadatas

def generate_followup_questions(query, answer, text_model_name): # Added text_model_name
     """Generates relevant follow-up questions using the specified LLM."""
     prompt = f"""Based on the user's question and the provided answer, suggest exactly 3 concise and relevant follow-up questions... (rest of prompt same as before) ...

     Original Question: {query}
     Provided Answer: {answer}

     Suggested Follow-up Questions (Python list format ONLY):
     """
     try:
         # *** Use the passed model name ***
         model = genai.GenerativeModel(text_model_name)
         generation_config = genai.types.GenerationConfig(temperature=0.4)
         response = model.generate_content(prompt, generation_config=generation_config)

         if hasattr(response, 'text'):
             # ... (parsing logic same as before) ...
             response_text = response.text.strip()
             log_message(f"Raw follow-up suggestions: {response_text}", "debug")
             match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', response_text, re.DOTALL)
             if match:
                  list_str = match.group()
                  try:
                       import ast
                       followups = ast.literal_eval(list_str)
                       if isinstance(followups, list) and all(isinstance(q, str) for q in followups):
                            log_message(f"Parsed follow-ups via ast: {followups}", "debug")
                            return followups[:3]
                  except Exception as eval_err:
                       log_message(f"ast.literal_eval failed for follow-up list '{list_str}': {eval_err}", "warning")

             lines = [line.strip(' -*\'" ') for line in response_text.split('\n')]
             questions = [line for line in lines if line.endswith('?') and len(line) > 10]
             if questions:
                  log_message(f"Fallback parsed follow-ups (lines): {questions[:3]}", "debug")
                  return questions[:3]

             log_message("Could not reliably parse follow-up suggestions into a list.", "warning")
             return []
         else:
              log_message("LLM response for follow-ups has no text part.", "warning")
              return []
     except Exception as e:
         st.exception(e)
         log_message(f"Error generating follow-up questions with LLM {text_model_name}: {e}", "error")
         return []

# --- Need get_current_chat_state if answer_data_query uses it ---
# This is problematic design. It's better to pass the data_frames dict.
# For now, adding it here to avoid NameError, but this indicates tight coupling.
def get_current_chat_state():
    """Placeholder - Needs proper implementation or removal of dependency."""
    log_message("Warning: qa_engine calling get_current_chat_state - tight coupling.", "warning")
    if "current_chat_id" in st.session_state and st.session_state.current_chat_id in st.session_state.get("chats", {}):
         return st.session_state.chats[st.session_state.current_chat_id]
    return None

