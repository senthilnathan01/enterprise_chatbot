# image_processor.py
"""Handles image processing and interaction with Vision Language Models."""

import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
from utils import log_message
from config import VISION_MODEL_NAME

# Ensure API key is configured before calling model functions
# This function assumes genai is already configured in the main app


def get_gemini_vision_description_ocr(image_bytes, filename="image"):
    """
    Gets description and attempts OCR using Gemini Vision.
    Assumes genai is configured.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        model = genai.GenerativeModel(VISION_MODEL_NAME)
        # Ask for both description and text extraction
        response = model.generate_content([
            "Describe this image and extract any text visible within it. Format the output clearly distinguishing description from extracted text.",
             img
        ])
        if response.parts:
            return response.text # Contains both description and OCR'd text
        elif response.prompt_feedback.block_reason:
            log_message(f"Image processing for {filename} blocked: {response.prompt_feedback.block_reason}", "warning")
            return f"[Image: {filename} - Processing Blocked: {response.prompt_feedback.block_reason}]"
        else:
            log_message(f"Could not generate description/OCR for {filename}.", "warning")
            return f"[Image: {filename} - No description or text extracted]"
    except Exception as e:
        # Log specific GenAI errors if possible
        log_message(f"Error processing image {filename} with Gemini Vision: {e}", "error")
        return f"[Image: {filename} - Error processing image: {e}]"