# app.py
import streamlit as st
import os
from data_ingestion.data_ingestion import ingest_files

def save_uploaded_file(uploaded_file, save_dir="uploaded_files"):
    """Save an uploaded file to a folder on disk."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("Data Ingestion for Emotional AI")
    st.write("Upload your documents here to train the language model.")

    # Accept multiple files of certain types
    uploaded_files = st.file_uploader("Choose your files", 
                                      accept_multiple_files=True, 
                                      type=["pdf", "docx", "txt"])

    # Once user has uploaded files
    if st.button("Ingest Files"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return

        # Save each file and build list of file paths
        file_paths = []
        for uploaded_file in uploaded_files:
            saved_path = save_uploaded_file(uploaded_file)
            file_paths.append(saved_path)
        
        # Call the ingestion script
        output_path = "combined_corpus.txt"
        ingest_files(file_paths, output_path=output_path)
        
        st.success(f"Files have been ingested and combined into {output_path}!")

if __name__ == "__main__":
    main()
