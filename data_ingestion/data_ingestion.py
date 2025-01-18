import logging
import os
import re
import pdfplumber
import docx
from concurrent.futures import ProcessPoolExecutor  # or ThreadPoolExecutor
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def set_logging_level(level=logging.INFO):
    """
    Adjust logging verbosity. For example:
        set_logging_level(logging.WARNING)
        set_logging_level(logging.DEBUG)
    """
    logging.getLogger().setLevel(level)


def read_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
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
    """
    Extract text from a DOCX file using python-docx.
    """
    try:
        document = docx.Document(docx_path)
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs)
    except Exception as e:
        logging.error(f"Error reading DOCX '{docx_path}': {e}")
        return ""


def read_txt(txt_path):
    """
    Simple function to read a .txt file.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading TXT '{txt_path}': {e}")
        return ""


def domain_specific_clean(text):
    """
    Optionally, add additional mental-health-specific cleaning or
    anonymization steps. Right now, it's just a placeholder that returns text.
    """
    # Example:
    # text = re.sub(r"PATIENT:\s*[A-Z0-9]+", "[REDACTED PATIENT]", text)
    return text


def clean_text(text):
    """
    Remove URLs and multiple spaces/newlines.
    Adjust as needed for your domain (e.g., remove punctuation, special chars, etc.).
    Also calls domain_specific_clean for mental-health text.
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Potential additional domain-specific cleaning
    text = domain_specific_clean(text)

    return text


def read_and_clean_file(path):
    """
    Read and clean a file based on its extension. Returns the cleaned text.
    Handles exceptions gracefully so a single bad file won't crash the entire process.
    """
    if not os.path.isfile(path):
        logging.warning(f"Skipping non-existent file: {path}")
        return ""

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        raw_text = read_pdf(path)
    elif ext == ".docx":
        raw_text = read_docx(path)
    elif ext == ".txt":
        raw_text = read_txt(path)
    else:
        logging.warning(f"Skipping unknown file type: {path}")
        return ""

    return clean_text(raw_text)


def ingest_files(file_paths, output_path="combined_corpus.txt", max_workers=None):
    """
    Reads multiple file types (PDF, DOCX, TXT), cleans the text,
    then writes a combined .txt file at `output_path`.

    By default uses ProcessPoolExecutor to handle CPU-bound tasks
    (like parsing PDFs). If you find it's more I/O-bound, switch back to
    ThreadPoolExecutor.

    If `max_workers` is None, we auto-determine a suitable number based on CPU cores.
    """
    # Filter out non-existent paths
    valid_paths = [fp for fp in file_paths if os.path.isfile(fp)]
    if not valid_paths:
        logging.error("No valid file paths found. Exiting ingestion.")
        return

    if max_workers is None:
        # Example heuristic: up to (num_cores + 4) but capped at 32
        max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)

    logging.info(f"Starting ingestion of {len(valid_paths)} files with {max_workers} workers...")

    # We open the output file once and stream text line by line as it's processed
    try:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            # Using a local helper to process each file and write immediately
            def process_path(path):
                text = read_and_clean_file(path)
                if text:
                    out_f.write(text + "\n")

            # Use ProcessPoolExecutor for potentially CPU-heavy PDF parsing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_path, valid_paths)

        logging.info(f"Combined cleaned text saved to {output_path}")
    except Exception as e:
        logging.error(f"Could not write to output file '{output_path}': {e}")


# Example usage if running this file directly
if __name__ == "__main__":
    set_logging_level(logging.INFO)  # or logging.WARNING/DEBUG, etc.

    sample_files = [
        "sample1.pdf",
        "sample2.docx",
        "mental_health_notes.txt"
    ]
    ingest_files(sample_files, output_path="combined_corpus.txt")

