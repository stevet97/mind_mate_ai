import logging
import os
import pdfplumber
import docx
import re
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    import pdfplumber
except ImportError:
    raise ImportError("Please install pdfplumber to process PDF files: pip install pdfplumber")

try:
    import docx
except ImportError:
    raise ImportError("Please install python-docx to process DOCX files: pip install python-docx")


def read_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
    text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        logging.error(f"Error reading PDF '{pdf_path}': {e}")
        return ""
    return "\n".join(text)


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


def clean_text(text):
    """
    Remove URLs and multiple spaces/newlines.
    Adjust as needed for your domain (e.g., remove punctuation, special chars, etc.).
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_and_clean_file(path):
    """
    Read and clean a file based on its extension. Returns the cleaned text.
    Handles exceptions gracefully so a single bad file won't crash the entire process.
    """
    if not os.path.isfile(path):
        logging.warning(f"Skipping non-existent file: {path}")
        return None

    ext = os.path.splitext(path)[1].lower()
    raw_text = ""
    if ext == ".pdf":
        raw_text = read_pdf(path)
    elif ext == ".docx":
        raw_text = read_docx(path)
    elif ext == ".txt":
        raw_text = read_txt(path)
    else:
        logging.warning(f"Skipping unknown file type: {path}")
        return None

    return clean_text(raw_text)


def ingest_files(file_paths, output_path="combined_corpus.txt", max_workers=4):
    """
    Reads multiple file types (PDF, DOCX, TXT), cleans the text,
    then writes a combined .txt file at `output_path`.
    Uses ThreadPoolExecutor for concurrency to process files in parallel.
    """
    # Filter out non-existent paths beforehand
    valid_paths = [fp for fp in file_paths if os.path.isfile(fp)]
    if not valid_paths:
        logging.error("No valid file paths found. Exiting ingestion.")
        return

    logging.info(f"Starting ingestion of {len(valid_paths)} files...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each path to the read_and_clean_file function
        futures = {executor.submit(read_and_clean_file, path): path for path in valid_paths}
        for future in futures:
            # Gather results as they complete
            result = future.result()
            if result:
                results.append(result)

    # Combine all cleaned text
    combined_text = "\n".join(results)
    try:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(combined_text)
        logging.info(f"Combined cleaned text saved to {output_path}")
    except Exception as e:
        logging.error(f"Could not write to output file '{output_path}': {e}")


if __name__ == "__main__":
    # Example usage:
    sample_files = [
        "sample1.pdf",
        "sample2.docx",
        "mental_health_notes.txt"
    ]
    ingest_files(sample_files, output_path="combined_corpus.txt", max_workers=4)


#
