import logging
import os
import re
import pdfplumber
import docx
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Import your get_toxicity_score function from toxic_filter
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
    # Add any mental-health-specific or domain-specific cleaning here if needed
    return text

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    # Domain-specific
    text = domain_specific_clean(text)
    return text

def process_file(path):
    """
    Returns a dict:
    {
        'filename': <str>,
        'cleaned_text': <str>,
        'toxicity': <float>
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

    # Clean
    cleaned = clean_text(raw_text)

    # Get toxicity score, forcing it to float
    tox_score = get_toxicity_score(cleaned, max_length=1000)
    if tox_score is None:
        tox_score = 0.0
    else:
        # ensure numeric
        tox_score = float(tox_score)

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
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for file_info in executor.map(process_file, valid_paths):
                # If process_file returns None for any reason, replace with a default dict
                if not file_info or not isinstance(file_info, dict):
                    file_info = {
                        "filename": "unknown",
                        "cleaned_text": "",
                        "toxicity": 0.0
                    }
                # Append to results
                results.append(file_info)

        # Write to disk
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for r in results:
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
