import streamlit as st
import os
from data_ingestion.data_ingestion import ingest_files

def save_uploaded_file(uploaded_file, save_dir="uploaded_files"):
    """
    Save an uploaded file to a folder on disk.
    The 'uploaded_files' folder holds the actual files
    you drag and drop.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    # -- SIDEBAR --
    st.sidebar.title("Instructions")
    st.sidebar.write(
        "1. Select one or more files (PDF, DOCX, or TXT).\n"
        "2. Click **Ingest Files**.\n"
        "3. A combined_corpus.txt file is created from your uploads.\n"
        "4. Optionally, download the combined file using the button."
    )

    st.title("Data Ingestion for Emotional AI")
    st.write("Upload your documents here to train the language model.")
    
    # Accept multiple files of certain types
    uploaded_files = st.file_uploader(
        "Choose your files", 
        accept_multiple_files=True, 
        type=["pdf", "docx", "txt"]
    )

    # If user has uploaded some files, show them in a list
    if uploaded_files:
        st.subheader("Files ready to be ingested:")
        for uf in uploaded_files:
            st.write(f"- {uf.name}")
    else:
        st.info("No files uploaded yet. Upload some to begin.")

    if st.button("Ingest Files"):
        # If user clicks ingest but hasn't uploaded anything
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return

        # Save each file and build list of file paths
        file_paths = []
        for uploaded_file in uploaded_files:
            saved_path = save_uploaded_file(uploaded_file)
            file_paths.append(saved_path)

        # Show a spinner/progress indicator while ingesting
        with st.spinner("Ingesting files..."):
            output_path = "combined_corpus.txt"
            ingest_files(file_paths, output_path=output_path)

        st.success(
            f"Ingested {len(file_paths)} file(s) and created/updated '{output_path}'!"
        )

        # Add a download button to retrieve the combined_corpus.txt
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download combined_corpus.txt",
                    data=f,
                    file_name="combined_corpus.txt",
                    mime="text/plain"
                )
        else:
            st.error(
                "combined_corpus.txt was not found. "
                "Please check if the ingestion script created it properly."
            )
    
    # Provide a note on where ingested files go
    st.write("---")
    st.write(
        "Note: All uploaded files are stored in the `uploaded_files/` folder "
        "on the server running this app."
    )


if __name__ == "__main__":
    main()
