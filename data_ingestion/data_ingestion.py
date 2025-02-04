import logging
import os
import re
import docx
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader  # Using PyPDF for better PDF parsing
from collections import Counter

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

def extract_text(file_path):
    """Extracts text from PDF, DOCX, or TXT files while preserving structure."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            texts = [page.extract_text().strip() for page in reader.pages if page.extract_text()]
            extracted_text = "\n".join(texts)
        elif ext == ".docx":
            document = docx.Document(file_path)
            extracted_text = "\n".join([p.text.strip() for p in document.paragraphs if p.text.strip()])
        elif ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        else:
            logging.warning(f"Skipping unsupported file type: {file_path}")
            return None

        if not extracted_text.strip():
            logging.warning(f"⚠️ No valid text extracted from: {file_path}")
            return None

        return clean_text(extracted_text)

    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return None

def process_file(file_path):
    """Processes a file, extracts text, and applies toxicity filtering."""
    try:
        if not os.path.isfile(file_path):
            logging.warning(f"Skipping non-existent file: {file_path}")
            return None

        raw_text = extract_text(file_path)
        if not raw_text:
            return None

        # Ensure we do not exceed processing limits
        truncated_text = raw_text[:2000]
        toxicity_score = get_toxicity_score(truncated_text) or 0.0

        return {
            "filename": os.path.basename(file_path),
            "cleaned_text": truncated_text,
            "toxicity": round(float(toxicity_score), 4)
        }

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def ingest_files(file_paths, output_path="cleaned_corpus.txt", max_workers=4, skip_toxic=True, toxicity_threshold=0.5):
    """Ingests files, processes them in parallel, and saves cleaned text."""
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
                result = future.result(timeout=15)
                if result:
                    logging.info(f"✅ Processed: {result['filename']} | Text Length: {len(result['cleaned_text'])} | Toxicity: {result['toxicity']}")
                    results.append(result)
                else:
                    logging.warning(f"⚠️ No valid text extracted from: {future_to_file[future]}")
            except Exception as e:
                logging.error(f"Worker failed on {future_to_file[future]}: {e}")
                
    # Filter results based on toxicity
    filtered_results = [r for r in results if r and (not skip_toxic or r["toxicity"] < toxicity_threshold)]

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in filtered_results:
                f.write(f"=== File: {item['filename']} (Toxicity: {item['toxicity']}) ===\n")
                f.write(item["cleaned_text"] + "\n\n")
        logging.info(f"✅ Saved cleaned text to {output_path} ({len(filtered_results)} files included)")
    except Exception as write_err:
        logging.error(f"Failed to write output file: {write_err}")

    return filtered_results


