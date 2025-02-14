import logging
import os
import re
import time
import json
import docx
import pytesseract
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
from collections import Counter
from pdf2image import convert_from_path
from time import sleep

# Import toxicity filter
from toxic_filter.toxic_filter import get_toxicity_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_text(text):
    """Removes unwanted elements like extra spaces, URLs, timestamps, legal disclaimers, and broken formatting."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}', '', text)  # Remove timestamps
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = re.sub(r'([-*])\s+', '\n• ', text)  # Fix bullet points
    text = re.sub(r'\(Link:\s*(.*?)\)', r' [→ \1]', text)  # Fix hyperlinks
    
    # Remove unwanted navigation text and policies
    text = re.sub(r'\b(privacy policy|cookie policy|terms of use|contact us|site map|all rights reserved)\b.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(Home|Menu|Search|Forum|Helpline|Advice|How we can help|Get involved|Forum).*$', '', text, flags=re.MULTILINE)
    
    # Fix encoding artifacts
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    
    return text

def extract_text(file_path, use_ocr=False):
    """
    Extracts text from PDF, DOCX, or TXT files while preserving structure.
    """
    ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            texts = [page.extract_text() for page in reader.pages if page.extract_text()]
            extracted_text = "\n".join(texts)

            # Use OCR if no text found
            if use_ocr and not extracted_text.strip():
                logging.warning(f"⚠️ No text found in {file_path}. Using OCR...")
                images = convert_from_path(file_path)
                extracted_text = "\n".join([pytesseract.image_to_string(img) for img in images])

        elif ext == ".docx":
            document = docx.Document(file_path)
            extracted_text = "\n".join([p.text.strip() for p in document.paragraphs if p.text.strip()])

        elif ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        
        if not extracted_text.strip():
            logging.warning(f"⚠️ No valid text extracted from: {file_path}")
            return None

        return clean_text(extracted_text)
    except Exception as e:
        logging.error(f"❌ Error extracting text from {file_path}: {e}")
        return None

def process_file(file_path, max_file_size_mb=10, timeout=30):
    """Processes a file, extracts text, and applies toxicity filtering."""
    try:
        if not os.path.isfile(file_path):
            logging.warning(f"Skipping non-existent file: {file_path}")
            return None

        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        if file_size > max_file_size_mb:
            logging.warning(f"⚠️ Skipping large file: {file_path} ({file_size:.2f}MB)")
            return None

        start_time = time.time()
        raw_text = extract_text(file_path)
        if not raw_text:
            return None

        truncated_text = raw_text[:2000]
        toxicity_score = get_toxicity_score(truncated_text) or 0.0
        elapsed_time = round(time.time() - start_time, 2)

        logging.info(f"✅ Processed: {os.path.basename(file_path)} | Length: {len(truncated_text)} chars | Toxicity: {toxicity_score} | Time: {elapsed_time}s")
        return {
            "filename": os.path.basename(file_path),
            "cleaned_text": truncated_text,
            "toxicity": round(float(toxicity_score), 4)
        }
    except Exception as e:
        logging.error(f"❌ Error processing file {file_path}: {e}")
        return None

def ingest_files(file_paths, output_path="cleaned_corpus.jsonl", max_workers=4, skip_toxic=True, toxicity_threshold=0.5):
    """Ingests files, processes them in parallel, and saves cleaned text in JSONL format."""
    valid_paths = [fp for fp in file_paths if os.path.isfile(fp)]
    if not valid_paths:
        logging.error("No valid file paths found. Exiting ingestion.")
        return []

    logging.info(f"Processing {len(valid_paths)} files with {max_workers} workers...")
    results, failed_files = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path in valid_paths:
            future = executor.submit(process_file, path)
            try:
                result = future.result(timeout=20)
                if result:
                    results.append(result)
                else:
                    failed_files.append(path)
            except Exception as e:
                logging.error(f"❌ Error processing {path}: {e}")
                failed_files.append(path)

    # Filter results based on toxicity
    filtered_results = [r for r in results if r and (not skip_toxic or r["toxicity"] < toxicity_threshold)]

    # Save output in JSONL format
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in filtered_results:
                json.dump({"text": item["cleaned_text"], "source": item["filename"], "toxicity": item["toxicity"]}, f)
                f.write("\n")
        logging.info(f"✅ Saved cleaned text to {output_path} ({len(filtered_results)} files included)")
    except Exception as write_err:
        logging.error(f"Failed to write output file: {write_err}")

    if failed_files:
        logging.warning(f"⚠️ {len(failed_files)} files failed after all retry attempts: {failed_files}")

    return filtered_results






