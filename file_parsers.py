# file_parsers.py
"""Functions to parse different file types, extracting text, images, data, and URLs."""

import streamlit as st
import io
import fitz # PyMuPDF
import docx # python-docx
from pptx import Presentation # python-pptx
import pandas as pd
import json
from utils import log_message, find_urls
from image_processor import get_gemini_vision_description_ocr # Import image processor

# --- PDF Parser ---
def parse_pdf(file_content, filename):
    text = ""
    images_analysis = [] # Store text from image analysis/OCR
    urls = set()
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        log_message(f"Processing PDF '{filename}': {len(doc)} pages.", "info")
        for page_num, page in enumerate(doc):
            page_text = ""
            try:
                page_text = page.get_text()
                urls.update(find_urls(page_text))
            except Exception as text_err:
                 log_message(f"Error extracting text from {filename} page {page_num+1}: {text_err}", "warning")

            # Enhanced OCR: Check if text is minimal, fallback to OCR if needed
            # Heuristic: Less than 50 chars might indicate an image-based page or sparse text
            ocr_attempted = False
            if len(page_text.strip()) < 50:
                 log_message(f"Page {page_num+1} in '{filename}' has little text, attempting OCR.", "info")
                 ocr_attempted = True
                 try:
                     # Render page to image for OCR
                     pix = page.get_pixmap(dpi=150) # Control resolution for OCR
                     img_bytes = pix.tobytes("png")
                     ocr_text = get_gemini_vision_description_ocr(img_bytes, f"{filename}_page{page_num+1}_ocr")
                     page_text += "\n\n[Page Image OCR Result:]\n" + ocr_text + "\n"
                     images_analysis.append(ocr_text)
                 except Exception as ocr_err:
                      log_message(f"Error during OCR fallback for {filename} page {page_num+1}: {ocr_err}", "warning")

            text += page_text + "\n\n" # Add page separator

            # Extract embedded images (only if OCR wasn't already run on the whole page)
            if not ocr_attempted:
                 try:
                     image_list = page.get_images(full=True)
                     for img_index, img_info in enumerate(image_list):
                          xref = img_info[0]
                          base_image = doc.extract_image(xref)
                          if base_image: # Check if image extraction was successful
                              image_bytes = base_image["image"]
                              img_meta_filename = f"{filename}_page{page_num+1}_img{img_index}"
                              # Use st.spinner for long operations if running interactively?
                              # In batch processing, spinner doesn't show well. Logging is key.
                              # log_message(f"Analyzing embedded image {img_meta_filename}...", "debug")
                              img_desc_ocr = get_gemini_vision_description_ocr(image_bytes, img_meta_filename)
                              text += f"\n[Embedded Image: {img_meta_filename} - Analysis:\n{img_desc_ocr}]\n\n"
                              images_analysis.append(img_desc_ocr)
                          else:
                               log_message(f"Could not extract embedded image xref {xref} on {filename} page {page_num+1}.", "warning")
                 except Exception as img_err:
                      log_message(f"Error processing embedded images on {filename} page {page_num+1}: {img_err}", "warning")

        log_message(f"Finished processing PDF '{filename}'. Found URLs: {len(urls)}")
    except Exception as e:
        log_message(f"Critical error processing PDF '{filename}': {e}", "error")
        return "", [], [] # Return empty on critical failure
    return text, images_analysis, list(urls)


# --- DOCX Parser ---
def parse_docx(file_content, filename):
    text = ""
    images_analysis = []
    urls = set()
    try:
        doc_stream = io.BytesIO(file_content)
        doc = docx.Document(doc_stream)
        log_message(f"Processing DOCX '{filename}'.", "info")

        full_text = []
        for para in doc.paragraphs:
            para_text = para.text
            full_text.append(para_text)
            urls.update(find_urls(para_text))

        # Handle hyperlinks explicitly
        for rel in doc.part.rels.values():
            if rel.reltype == docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK and rel.is_external:
                urls.add(rel.target_ref)

        text = '\n\n'.join(full_text)

        # Image extraction
        image_parts = {}
        for rel_id, rel in doc.part.rels.items():
             if "image" in rel.target_ref:
                 try:
                     image_parts[rel_id] = rel.target_part.blob
                 except Exception as img_access_err:
                      log_message(f"Could not access image data for rel {rel_id} in {filename}: {img_access_err}", "warning")


        img_count = 0
        # Check inline shapes and potentially other image types
        for shape in doc.inline_shapes:
             # This part needs more refinement based on OOXML structure for reliable extraction
             # The following is a simplified check based on common elements
             blip_element = shape._inline.graphic.graphicData.pic.blipFill.blip
             if blip_element is not None:
                  embed_id = blip_element.embed
                  if embed_id in image_parts:
                       image_bytes = image_parts[embed_id]
                       img_meta_filename = f"{filename}_img{img_count}"
                       # log_message(f"Analyzing image {img_meta_filename} from DOCX...", "debug")
                       img_desc_ocr = get_gemini_vision_description_ocr(image_bytes, img_meta_filename)
                       text += f"\n\n[Embedded Image: {img_meta_filename} - Analysis:\n{img_desc_ocr}]\n\n"
                       images_analysis.append(img_desc_ocr)
                       img_count += 1
                       del image_parts[embed_id] # Avoid reprocessing

        log_message(f"Finished processing DOCX '{filename}'. Found URLs: {len(urls)}, Images processed: {img_count}")
    except Exception as e:
        log_message(f"Critical error processing DOCX '{filename}': {e}", "error")
        return "", [], []
    return text, images_analysis, list(urls)

# --- PPTX Parser ---
def parse_pptx(file_content, filename):
    text = ""
    images_analysis = []
    urls = set()
    try:
        ppt_stream = io.BytesIO(file_content)
        prs = Presentation(ppt_stream)
        log_message(f"Processing PPTX '{filename}': {len(prs.slides)} slides.", "info")

        for i, slide in enumerate(prs.slides):
            slide_text = f"\n--- Slide {i+1} ---\n"
            slide_urls = set()
            try:
                for shape in slide.shapes:
                    shape_text_content = ""
                    if shape.has_text_frame:
                         shape_text_content = shape.text_frame.text
                         slide_text += shape_text_content + "\n"
                         slide_urls.update(find_urls(shape_text_content))

                    if shape.has_table:
                        table_text = "\n[Table Data:]\n"
                        try:
                            for row in shape.table.rows:
                                row_data = [cell.text_frame.text for cell in row.cells]
                                table_text += "| " + " | ".join(row_data) + " |\n"
                            slide_text += table_text
                        except Exception as table_err:
                             log_message(f"Error reading table on slide {i+1} in {filename}: {table_err}", "warning")

                    # Handle hyperlinks associated with shapes or text runs
                    if shape.click_action.hyperlink.address:
                        slide_urls.add(shape.click_action.hyperlink.address)
                    if hasattr(shape, 'text_frame'):
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                if run.hyperlink.address:
                                    slide_urls.add(run.hyperlink.address)

                    # Handle images
                    # pptx library makes accessing image bytes slightly less direct
                    # This requires iterating through image parts and matching
                    if shape.shape_type == 13: # MSO_SHAPE_TYPE.PICTURE
                        try:
                             image = shape.image
                             image_bytes = image.blob
                             img_meta_filename = f"{filename}_slide{i+1}_img{shape.shape_id}"
                             # log_message(f"Analyzing image {img_meta_filename} from PPTX...", "debug")
                             img_desc_ocr = get_gemini_vision_description_ocr(image_bytes, img_meta_filename)
                             slide_text += f"\n[Embedded Image: {img_meta_filename} - Analysis:\n{img_desc_ocr}]\n\n"
                             images_analysis.append(img_desc_ocr)
                        except Exception as img_err:
                             # Shape might be picture placeholder without image, etc.
                             log_message(f"Could not process shape as image on slide {i+1} in {filename} (ID {shape.shape_id}): {img_err}", "warning")

            except Exception as shape_err:
                 log_message(f"Error processing shapes on slide {i+1} in {filename}: {shape_err}", "warning")

            text += slide_text
            urls.update(slide_urls)

        log_message(f"Finished processing PPTX '{filename}'. Found URLs: {len(urls)}")
    except Exception as e:
        log_message(f"Critical error processing PPTX '{filename}': {e}", "error")
        return "", [], []
    return text, images_analysis, list(urls)


# --- Data File Parsers (CSV/XLSX/JSON) ---
def parse_data_file(file_content, filename):
    """Parses CSV, XLSX, or JSON into a DataFrame or dict and creates a summary."""
    text_summary = ""
    data_object = None # Will store DataFrame or dict/list
    file_type = filename.split('.')[-1].lower()

    try:
        log_message(f"Processing Data file '{filename}'.", "info")
        if file_type == "csv":
            # Try detecting encoding, fall back to utf-8 ignore
            try:
                 df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
            except UnicodeDecodeError:
                 log_message(f"UTF-8 decode failed for {filename}, trying latin-1.", "debug")
                 df = pd.read_csv(io.BytesIO(file_content), encoding='latin1')
            data_object = df
        elif file_type == "xlsx":
            df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            data_object = df
        elif file_type == "json":
            try:
                 data_object = json.loads(file_content.decode('utf-8'))
            except UnicodeDecodeError:
                 log_message(f"UTF-8 decode failed for {filename}, trying latin-1.", "debug")
                 data_object = json.loads(file_content.decode('latin1'))


        if data_object is not None:
            # Store the actual data object in session state
            if "data_frames" not in st.session_state: st.session_state.data_frames = {}
            st.session_state.data_frames[filename] = data_object

            # Create a textual summary for indexing
            text_summary += f"--- Data Summary for {filename} ---\n"
            if isinstance(data_object, pd.DataFrame):
                text_summary += f"Type: Tabular Data (CSV/XLSX)\nShape: {data_object.shape}\n"
                buffer = io.StringIO()
                data_object.info(buf=buffer, verbose=True, show_counts=True) # More detailed info
                text_summary += "Columns and Data Types:\n" + buffer.getvalue() + "\n"
                text_summary += "First 5 Rows:\n" + data_object.head().to_string() + "\n"
                # Try basic statistics, handle potential errors if data is not numeric
                try:
                     text_summary += "Basic Statistics (numeric cols):\n" + data_object.describe().to_string() + "\n"
                     text_summary += "Basic Statistics (all cols):\n" + data_object.describe(include='all').to_string() + "\n"
                except Exception as desc_err:
                     log_message(f"Could not generate full statistics for {filename}: {desc_err}", "warning")
                     text_summary += "Basic Statistics: (Could not be fully generated)\n"
            elif isinstance(data_object, (dict, list)):
                 text_summary += f"Type: JSON Data ({type(data_object)})\n"
                 preview = json.dumps(data_object, indent=2)[:1500] # Limit preview size
                 text_summary += f"Data Preview:\n{preview}\n"
                 if len(preview) >= 1500: text_summary += "...\n" # Indicate truncation

            text_summary += f"--- End Data Summary for {filename} ---\n"
            log_message(f"Successfully parsed and summarized data from '{filename}'.")
        else:
            log_message(f"Could not parse data file '{filename}'.", "warning")
            return "", None # Return empty text, no data object

    except Exception as e:
        log_message(f"Error processing data file '{filename}': {e}", "error")
        return "", None
    # Return the summary text and the parsed data object
    return text_summary, data_object

# --- TXT Parser ---
def parse_txt(file_content, filename):
    text = ""
    urls = set()
    try:
        log_message(f"Processing TXT file '{filename}'.", "info")
        # Try common encodings
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            log_message(f"UTF-8 decode failed for {filename}, trying latin-1.", "debug")
            try:
                text = file_content.decode("latin1")
            except UnicodeDecodeError:
                 log_message(f"Latin-1 decode also failed for {filename}, using ignore.", "warning")
                 text = file_content.decode("utf-8", errors='ignore')

        urls.update(find_urls(text))
        log_message(f"Finished processing TXT '{filename}'. Found URLs: {len(urls)}")
    except Exception as e:
        log_message(f"Error processing TXT file '{filename}': {e}", "error")
        return "", [], []
    return text, [], list(urls) # No images expected

# --- Image File Parser ---
def parse_image(file_content, filename):
    text = ""
    images_analysis = []
    try:
        log_message(f"Processing Image file '{filename}'.", "info")
        # log_message(f"Analyzing image file {filename}...", "debug")
        img_desc_ocr = get_gemini_vision_description_ocr(file_content, filename)
        text = f"[Image File Content: {filename}]\n{img_desc_ocr}\n"
        images_analysis.append(img_desc_ocr)
        log_message(f"Finished processing image file: {filename}")
    except Exception as e:
        log_message(f"Error processing image file '{filename}': {e}", "error")
        return "", [], []
    return text, images_analysis, [] # No URLs expected