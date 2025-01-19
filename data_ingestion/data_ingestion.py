import logging
import os
import re
import pdfplumber
import docx
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from toxic_filter.toxic_filter import get_toxicity_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def set_logging_level(level=logging.INFO):
    logging.getLogger().setLevel(level)

def read_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        return "\n".join(texts)
    except Exception as e:
        logging.error(f"Error reading PDF '{pdf_path}': {e}")
        return ""

def read_docx(docx_path):
    try:
        document = docx.Document(docx_path)
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs)
    except Exception as e:
        logging.error(f"Error reading DOCX '{docx_path}': {e}")
        return ""

def read_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading TXT '{txt_path}': {e}")
        return ""

def domain_specific_clean(text):
    # Example: Additional domain cleaning steps if desired
    return text

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    text = domain_specific_clean(text)
    return text

def process_file(path):
    """
    Returns a dict:
    {
      'filename': <filename>,
      'cleaned_text': <string>,
      'toxicity': <float in [0,1]>
    }
    """
    if not os.path.isfile(path):
        logging.warning(f"Skipping non-existent file: {path}")
        return {
            "filename": os.path.basename(path),
            "cleaned_text": "",
            "toxicity": 0.0
        }

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        raw_text = read_pdf(path)
    elif ext == ".docx":
        raw_text = read_docx(path)
    elif ext == ".txt":
        raw_text = read_txt(path)
    else:
        logging.warning(f"Skipping unknown file type: {path}")
        return {
            "filename": os.path.basename(path),
            "cleaned_text": "",
            "toxicity": 0.0
        }

    cleaned = clean_text(raw_text)
    tox_score = get_toxicity_score(cleaned, max_length=1000)

    return {
        "filename": os.path.basename(path),
        "cleaned_text": cleaned,
        "toxicity": tox_score
    }

def ingest_files(file_paths, output_path="combined_corpus.txt", max_workers=None):
    """
    Reads multiple file types (PDF, DOCX, TXT) using multiprocessing,
    cleans the text, checks toxicity, then writes everything to a single output file.
    Returns a list of dicts: [{'filename':..., 'toxicity':..., 'cleaned_text':...}, ...]
    """
    valid_paths = [fp for fp in file_paths if os.path.isfile(fp)]
    if not valid_paths:
        logging.error("No valid file paths found. Exiting ingestion.")
        return []

    if max_workers is None:
        max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)

    logging.info(f"Starting ingestion of {len(valid_paths)} files with {max_workers} workers...")

    results = []
    try:
        # We gather file data in the parent after concurrency
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Each item is a dict with 'filename', 'cleaned_text', 'toxicity'
            for file_info in executor.map(process_file, valid_paths):
                results.append(file_info)

        # Write all text, including toxic, to the combined file
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for r in results:
                # Add a small header marking which file it came from (optional)
                out_f.write(f"=== File: {r['filename']} (Toxicity: {r['toxicity']:.2f}) ===\n")
                out_f.write(r["cleaned_text"] + "\n\n")

        logging.info(f"Combined cleaned text saved to {output_path}")
    except Exception as e:
        logging.error(f"Could not write to output file '{output_path}': {e}")

    return results

if __name__ == "__main__":
    set_logging_level(logging.INFO)
    sample_files = [
        "sample1.pdf",
        "sample2.docx",
        "mental_health_notes.txt"
    ]
    data = ingest_files(sample_files, output_path="combined_corpus.txt")
    print(data)  # e.g. see the toxicity metadata

