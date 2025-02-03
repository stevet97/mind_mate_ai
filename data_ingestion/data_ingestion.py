# data_ingestion/data_ingestion.py

import logging
import os
import re
import docx
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader  # Using PyPDF for better PDF parsing

# Import toxicity filter
from toxic_filter.toxic_filter import get_toxicity_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_text(text):
    """Removes unwanted elements like extra spaces, URLs, timestamps, and fixes bullet points and hyperlinks."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}', '', text)  # Remove timestamps
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces/newlines
    text = re.sub(r'(\*|-)\s+', '\n• ', text)  # Convert bullet points into a consistent format
    text = re.sub(r'\(Link:\s*(.*?)\)', r' [→ \1]', text)  # Fix broken hyperlinks
    return text

def extract_pdf_text(pdf_path):
    """Extracts text from a PDF file while preserving structure."""
    try:
        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text.strip())
        return clean_text("\n".join(texts))
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def extract_docx_text(docx_path):
    """Extracts text from a DOCX file while preserving formatting."""
    try:
        document = docx.Document(docx_path)
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        return clean_text("\n".join(paragraphs))
    except Exception as e:
        logging.error(f"Error extracting text from DOCX {docx_path}: {e}")
        return ""

def extract_txt_text(txt_path):
    """Reads text from a TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return clean_text(f.read())
    except Exception as e:
        logging.error(f"Error reading TXT {txt_path}: {e}")
        return ""

def process_file(file_path):
    """Processes a file, extracts text, cleans it, and computes toxicity."""
    try:
        if not os.path.isfile(file_path):
            logging.warning(f"Skipping non-existent file: {file_path}")
            return None
    
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            raw_text = extract_pdf_text(file_path)
        elif ext == ".docx":
            raw_text = extract_docx_text(file_path)
        elif ext == ".txt":
            raw_text = extract_txt_text(file_path)
        else:
            logging.warning(f"Skipping unsupported file type: {file_path}")
            return None
    
        if not raw_text.strip():
            logging.warning(f"Skipping empty file: {file_path}")
            return None
    
        cleaned_text = clean_text(raw_text)
    
        # Ensure we do not exceed model input limits
        truncated_text = cleaned_text[:2000]  
        toxicity_score = get_toxicity_score(truncated_text) or 0.0
    
        return {
            "filename": os.path.basename(file_path),
            "cleaned_text": cleaned_text,
            "toxicity": round(float(toxicity_score), 4)
        }
    
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None  # Ensures execution continues even if one file fails

def ingest_files(file_paths, output_path="cleaned_corpus.txt", max_workers=4, skip_toxic=True, toxicity_threshold=0.5):
    """Ingests files, processes them in parallel, and saves clean text."""
    valid_paths = [fp for fp in file_paths if os.path.isfile(fp)]
    if not valid_paths:
        logging.error("No valid file paths found. Exiting ingestion.")
        return []
    
    logging.info(f"Processing {len(valid_paths)} files with {max_workers} workers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, path): path for path in valid_paths}
    
        for future in future_to_file:
            try:
                result = future.result(timeout=15)  # Ensures workers do not hang indefinitely
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"Worker failed on {future_to_file[future]}: {e}")
    
    # Filter results based on toxicity
    filtered_results = [r for r in results if r and (not skip_toxic or r["toxicity"] < toxicity_threshold)]
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in filtered_results:
                f.write(f"=== File: {item['filename']} (Toxicity: {item['toxicity']}) ===\n")
                f.write(item["cleaned_text"] + "\n\n")
        logging.info(f"Saved cleaned text to {output_path} ({len(filtered_results)} files included)")
    except Exception as write_err:
        logging.error(f"Failed to write output file: {write_err}")
    
    return filtered_results
