# qa_engine.py
"""Handles the core question answering logic, including data queries and follow-ups."""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
from utils import log_message
from config import TEXT_MODEL_NAME

# Assumes genai is configured in main_app

def is_data_query(query):
    """Basic keyword-based check if a query seems related to structured data."""
    data_keywords = [
        "total", "average", "mean", "sum", "count", "list", "show me data",
        "maximum", "minimum", "value of", "in table", "from csv", "in excel",
        "json record", "how many rows", "column names", "what are the columns",
        "calculate", "statistics for", "summarize the data"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in data_keywords)

def answer_data_query(query, relevant_metadatas):
    """
    Attempts to answer a query using stored DataFrames or JSON data.
    Uses an LLM to interpret the question against the data description.
    """
    # Find potential data sources based on retrieved metadata (prefer 'data_summary' type)
    data_source_files = set()
    primary_sources = set(meta['source'] for meta in relevant_metadatas if meta.get('type') == 'data_summary')
    secondary_sources = set(meta['source'] for meta in relevant_metadatas if meta.get('source') and meta.get('source').split('.')[-1].lower() in ['csv', 'xlsx', 'json'])
    data_source_files.update(primary_sources)
    data_source_files.update(secondary_sources)


    if not data_source_files:
        log_message("Data query routing: No relevant data source file identified in context.", "debug")
        return None # Fallback to general RAG

    # Prioritize or let user choose if multiple sources? For now, use first found.
    # A better approach might format descriptions of all relevant sources for the LLM.
    target_file = list(data_source_files)[0]
    log_message(f"Attempting data query using source: {target_file}", "info")

    if "data_frames" not in st.session_state or target_file not in st.session_state.data_frames:
        log_message(f"DataFrame/JSON for {target_file} not found in session state.", "error")
        return f"Error: Could not find the data for {target_file} to answer the query." # Informative error

    data_content = st.session_state.data_frames[target_file]
    data_description = ""

    # Create description based on data type
    try:
        if isinstance(data_content, pd.DataFrame):
            data_description = f"You have access to a pandas DataFrame from the file '{target_file}'.\n"
            data_description += f"Columns and Data Types:\n{data_content.info(verbose=True)}\n" # Use info for details
            data_description += f"First 5 Rows:\n{data_content.head().to_string()}\n"
        elif isinstance(data_content, (dict, list)):
            data_description = f"You have access to JSON data from the file '{target_file}'.\n"
            data_description += f"Data Preview (up to 1000 chars):\n{json.dumps(data_content, indent=2)[:1000]}...\n"
        else:
            data_description = f"Data source '{target_file}' contains data of type: {type(data_content)}. Provide a basic summary or query based on its structure if possible."
    except Exception as desc_err:
         log_message(f"Error creating description for data query {target_file}: {desc_err}", "warning")
         data_description = f"Could not fully describe data from {target_file}. Attempt to answer based on available structure."


    # Use LLM to interpret question against data
    prompt = f"""You are a data analysis assistant. Your task is to answer the user's question based *strictly* on the provided data description and context.

    **Instructions:**
    1. Analyze the User Question and the Data Description.
    2. Determine if the question can be answered using *only* the provided data information (column names, types, preview, etc.).
    3. If the question asks for specific values, calculations (sum, average, count, min, max), or filtering based on the provided data:
        - State the calculation or filtering logic you would apply conceptually.
        - Provide the answer based on the data *if* it's directly visible or easily computable from the preview/description (e.g., count from shape, value from first 5 rows).
        - **Crucially: Do NOT invent data or perform complex calculations beyond what's visible or described.** State if the exact calculation requires accessing the full data, which you cannot do directly.
    4. If the question asks for general information *about* the data (e.g., "What are the columns?", "How many rows?"), answer based on the description.
    5. If the question cannot be answered from the provided description (e.g., requires complex joins, time series analysis, or data not shown), state clearly: "I cannot answer this question accurately from the provided data summary. Access to the full data and computational tools would be required."
    6. Be concise and precise. Do not guess or hallucinate.

    **Data Description:**
    ```
    {data_description}
    ```

    **User Question:**
    ```
    {query}
    ```

    **Answer:**
    """

    try:
        model = genai.GenerativeModel(TEXT_MODEL_NAME)
        # Use lower temperature for more factual data answers
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        response = model.generate_content(prompt, generation_config=generation_config)

        if response.parts:
            log_message("Generated answer using data query logic.", "info")
            return response.text # Return the LLM's interpretation/answer
        elif response.prompt_feedback.block_reason:
             log_message(f"Data query answer blocked: {response.prompt_feedback.block_reason}", "warning")
             return f"Could not answer data query due to safety settings: {response.prompt_feedback.block_reason}"
        else:
             log_message("LLM did not generate an answer for the data query.", "warning")
             return None # Fallback to general Q&A logic might be better here
    except Exception as e:
        st.exception(e)
        log_message(f"Error generating data query answer with LLM: {e}", "error")
        return f"An error occurred while trying to answer the data query: {e}"


def generate_answer_from_context(query, context_docs, context_metadatas):
    """Generates answer using LLM based purely on retrieved text context (RAG)."""
    if not context_docs:
        log_message("No relevant context found in documents for the query.", "warning")
        return "I cannot answer this question based on the provided documents.", []

    # Prepare context string with source information
    context_items = []
    source_files = set()
    for doc, meta in zip(context_docs, context_metadatas):
         source = meta.get('source', 'Unknown')
         source_files.add(source)
         context_items.append(f"Source: {source} (Chunk {meta.get('chunk_index', 'N/A')})\nContent:\n{doc}")
    context_str = "\n\n---\n\n".join(context_items)

    prompt = f"""You are a helpful assistant answering questions based *only* on the provided context from uploaded documents.

    **Instructions:**
    1. Carefully read the User Question and the Provided Context sections.
    2. Formulate a comprehensive answer to the question using *only* the information present in the context snippets.
    3. **Do not use any prior knowledge or information outside the provided context.**
    4. If the answer is explicitly stated in the context, quote or reference the key information.
    5. If the answer requires synthesizing information from multiple context snippets, combine them logically.
    6. Cite the source file(s) mentioned in the context (e.g., "According to 'report.pdf'...") when possible and relevant.
    7. If the context does not contain information to answer the question, state clearly: "I cannot answer this question based on the provided documents." or "The provided documents do not contain information about X."
    8. Ensure the answer is accurate, concise, and directly addresses the user's question. Avoid speculation.

    **Provided Context:**
    ```
    {context_str}
    ```

    **User Question:**
    ```
    {query}
    ```

    **Answer:**
    """

    try:
        model = genai.GenerativeModel(TEXT_MODEL_NAME)
        # Slightly higher temperature might allow for better synthesis
        generation_config = genai.types.GenerationConfig(temperature=0.5)
        response = model.generate_content(prompt, generation_config=generation_config)

        if response.parts:
            answer_text = response.text
            # Simple check to add source info if model didn't include it well
            if not any(f in answer_text for f in source_files) and len(source_files) < 4: # Add sources if not too many
                 answer_text += f"\n\n(Sources consulted: {', '.join(sorted(list(source_files)))})"

            return answer_text, context_metadatas # Return text answer and the context used
        elif response.prompt_feedback.block_reason:
             log_message(f"General answer generation blocked: {response.prompt_feedback.block_reason}", "warning")
             return f"Answer blocked due to safety settings: {response.prompt_feedback.block_reason}", context_metadatas
        else:
             log_message("LLM did not generate a general answer.", "warning")
             return "Sorry, I couldn't generate an answer based on the context.", context_metadatas
    except Exception as e:
        st.exception(e)
        log_message(f"Error generating general answer with Gemini: {e}", "error")
        return f"An error occurred while generating the answer: {e}", context_metadatas

def generate_followup_questions(query, answer):
     """Generates relevant follow-up questions using Gemini Pro."""
     # Use a precise prompt to get list format
     prompt = f"""Based on the user's question and the provided answer, suggest exactly 3 concise and relevant follow-up questions a user might ask next. The questions should logically extend the conversation or explore related details found in documents.

     Format the output ONLY as a Python list of strings, like this:
     ["Follow-up question 1?", "Follow-up question 2?", "Follow-up question 3?"]

     Original Question: {query}
     Provided Answer: {answer}

     Suggested Follow-up Questions (Python list format ONLY):
     """
     try:
         model = genai.GenerativeModel(TEXT_MODEL_NAME)
         # Lower temperature for more predictable formatting
         generation_config = genai.types.GenerationConfig(temperature=0.2)
         response = model.generate_content(prompt, generation_config=generation_config)

         if response.parts:
             response_text = response.text.strip()
             log_message(f"Raw follow-up suggestions: {response_text}", "debug")
             # Try to parse the list string safely using regex
             match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', response_text, re.DOTALL)
             if match:
                  list_str = match.group()
                  try:
                       followups = eval(list_str) # Use eval cautiously on the matched string
                       if isinstance(followups, list) and all(isinstance(q, str) for q in followups):
                            log_message(f"Parsed follow-ups: {followups}", "debug")
                            return followups[:3] # Limit to 3
                  except Exception as eval_err:
                       log_message(f"Eval failed for follow-up list '{list_str}': {eval_err}", "warning")

             # Fallback if regex/eval fails: look for lines ending in '?'
             lines = [line.strip(' -*') for line in response_text.split('\n')]
             questions = [line for line in lines if line.endswith('?') and len(line) > 5]
             if questions:
                  log_message(f"Fallback parsed follow-ups: {questions[:3]}", "debug")
                  return questions[:3]

             log_message("Could not reliably parse follow-up suggestions.", "warning")
             return [] # Could not parse structure
         else:
              log_message("LLM response for follow-ups has no parts.", "warning")
              return []
     except Exception as e:
         st.exception(e)
         log_message(f"Error generating follow-up questions: {e}", "error")
         return []