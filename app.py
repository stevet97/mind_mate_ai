# app.py
import streamlit as st
import os
import datetime
import logging

# google client libraries
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from data_ingestion.data_ingestion import ingest_files

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_drive_service():
    creds_dict = st.secrets["google_service_account"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials)

def upload_to_gdrive(local_file_path, folder_id="1fpMg9W19LF6YDJVjgUq2aylUqPfF1M"):
    service = get_drive_service()
    file_name = os.path.basename(local_file_path)
    file_metadata = {
        "name": file_name,
        "parents": [folder_id]
    }

    media = MediaFileUpload(local_file_path, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    return uploaded_file.get("id")

########################################
# The Streamlit App
########################################
def main():
    st.title("Data Ingestion & Google Drive Upload")
    st.write("Upload your documents here to train the language model.")

    GDRIVE_FOLDER_NAME = "My Drive / NOVA_project_data"
    st.info(f"Note: All ingested files will be uploaded to **{GDRIVE_FOLDER_NAME}**.")

    uploaded_files = st.file_uploader(
        "Choose your files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"]
    )

    if uploaded_files:
        st.subheader("Files ready to be ingested:")
        for uf in uploaded_files:
            st.write(f"- {uf.name}")
    else:
        st.info("No files uploaded yet. Upload some to begin.")

    if st.button("Ingest & Upload to Drive"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return

        file_paths = []
        with st.spinner("Saving uploaded files locally..."):
            os.makedirs("user_uploads", exist_ok=True)
            for uf in uploaded_files:
                saved_path = os.path.join("user_uploads", uf.name)
                with open(saved_path, "wb") as f:
                    f.write(uf.getbuffer())
                file_paths.append(saved_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"combined_corpus_{timestamp}.txt"

        with st.spinner("Ingesting files..."):
            # ingest_files now returns a list of dicts with 'filename', 'toxicity', etc.
            results = ingest_files(file_paths, output_path=output_path)

        # Now let's highlight any file above our threshold
        # Let's say 0.8 is quite high
        TOXICITY_THRESHOLD = 0.8
        high_toxic = [r for r in results if r['toxicity'] >= TOXICITY_THRESHOLD]

        st.subheader("Toxicity Summary:")
        if len(high_toxic) == 0:
            st.write("No files exceeded the 0.8 toxicity threshold.")
        else:
            for r in high_toxic:
                st.warning(
                    f"File **{r['filename']}** has high toxicity: {r['toxicity']*100:.1f}%"
                )

        # We still upload everything to Drive
        st.info(f"Uploading **{output_path}** to Google Drive folder: {GDRIVE_FOLDER_NAME}")

        folder_id = "1fpMg9W19LF6YDJVjgUq2aylUqPfF1M0R"
        try:
            upload_to_gdrive(output_path, folder_id=folder_id)
            st.success("Files have been ingested and uploaded to Google Drive successfully!")
            st.balloons()
        except Exception as e:
            st.error(f"Upload to Drive failed: {e}")
            return
        
        st.write("---")
        st.write("You can also download the combined_corpus.txt directly here:")
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download combined_corpus.txt",
                data=f,
                file_name="combined_corpus.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()

