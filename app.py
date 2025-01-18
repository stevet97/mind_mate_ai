import streamlit as st
import os

# 1) google client libraries
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 2) import concurrency ingestion logic from data_ingestion.py
from data_ingestion.data_ingestion import ingest_files

########################################
# authenticate with google drive
########################################
def get_drive_service():
    """
    Loads service account credentials from Streamlit secrets and
    returns a Google Drive API service resource.
    """
    # st.secrets["google_service_account"] should be a dict with your JSON data
    creds_dict = st.secrets["google_service_account"]

    credentials = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=credentials)
    return service



def upload_to_gdrive(local_file_path, folder_id="1fpMg9W19LF6YDJVjgUq2aylUqPfF1M"):
    """
    Uploads a local file to Google Drive in the folder:
      'My Drive / NOVA_project_data' (ID: 1fpMg9W19LF6YDJVjgUq2aylUqPfF1M).
    
    Using the googleapiclient library for the Drive API.
    Adjust as needed for your environment.
    """
    service = get_drive_service()

    file_name = os.path.basename(local_file_path)  # e.g. "combined_corpus.txt"
    file_metadata = {
        "name": file_name,
        "parents": [folder_id]  # This places the file in the desired folder
    }

    media = MediaFileUpload(local_file_path, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    return uploaded_file.get("id")

##############################
# The Streamlit App
##############################

def main():
    st.title("Data Ingestion & Google Drive Upload")
    st.write("Upload your documents here to train the language model.")
    
    # Updated folder display name
    GDRIVE_FOLDER_NAME = "My Drive / NOVA_project_data"
    st.info(f"Note: All ingested files will be uploaded to **{GDRIVE_FOLDER_NAME}**.")

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

    # Let user click a button to ingest and upload
    if st.button("Ingest & Upload to Drive"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return

        # 1) Save files locally first
        file_paths = []
        with st.spinner("Saving uploaded files locally..."):
            os.makedirs("user_uploads", exist_ok=True)
            for uf in uploaded_files:
                saved_path = os.path.join("user_uploads", uf.name)
                with open(saved_path, "wb") as f:
                    f.write(uf.getbuffer())
                file_paths.append(saved_path)

        # 2) Ingest them into combined_corpus.txt
        output_path = "combined_corpus.txt"
        with st.spinner("Ingesting files..."):
            ingest_files(file_paths, output_path=output_path)
        
        # 3) Announce we’re uploading to Google Drive
        st.info(f"Uploading **{output_path}** to Google Drive folder: {GDRIVE_FOLDER_NAME}")

        # 4) Actually upload to Drive
        folder_id = "1fpMg9W19LF6YDJVjgUq2aylUqPfF1M0R"  
        try:
            upload_to_gdrive(output_path, folder_id=folder_id)
            st.success("Files have been ingested and uploaded to Google Drive successfully!")
            st.balloons()  # A fun Streamlit effect
        except Exception as e:
            st.error(f"Upload to Drive failed: {e}")
            return
        
        # 5) Provide a download button for the combined corpus if you like
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


