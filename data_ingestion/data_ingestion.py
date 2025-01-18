import logging
import os
import re
import pdfplumber
import docx
from concurrent.futures import ProcessPoolExecutor  # or ThreadPoolExecutor
import multiprocessing

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
    # Add any extra steps unique to mental-health data.
    return text


def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Potential domain-specific steps
    text = domain_specific_clean(text)

    return text


def read_and_clean_file(path):
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
    Reads multiple file types (PDF, DOCX, TXT) using multiprocessing,
    cleans the text, then writes everything to a single output file.

    We gather the processed text in the parent process (to avoid
    file-handle issues) and then write it all in one pass.
    """
    valid_paths = [fp for fp in file_paths if os.path.isfile(fp)]
    if not valid_paths:
        logging.error("No valid file paths found. Exiting ingestion.")
        return

    if max_workers is None:
        max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)

    logging.info(f"Starting ingestion of {len(valid_paths)} files with {max_workers} workers...")

    # Use a list to collect processed texts (avoid streaming in child processes)
    processed_texts = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # read_and_clean_file returns a string
            results = executor.map(read_and_clean_file, valid_paths)

            # Gather results in the parent
            for text in results:
                if text:
                    processed_texts.append(text)

        # Now write the combined result in the parent process
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for txt in processed_texts:
                out_f.write(txt + "\n")

        logging.info(f"Combined cleaned text saved to {output_path}")
    except Exception as e:
        logging.error(f"Could not write to output file '{output_path}': {e}")


if __name__ == "__main__":
    set_logging_level(logging.INFO)

    sample_files = [
        "sample1.pdf",
        "sample2.docx",
        "mental_health_notes.txt"
    ]
    ingest_files(sample_files, output_path="combined_corpus.txt")


