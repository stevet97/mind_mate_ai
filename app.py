import streamlit as st
import os
import datetime
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from data_ingestion.data_ingestion import ingest_files

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

########################################
# Constants / Config
########################################
TOXICITY_THRESHOLD = 0.5
GDRIVE_FOLDER_NAME = "My Drive / NOVA_project_data"
FOLDER_ID = "1fpMg9W19LF6YDJVjgUq2aylUqPfF1M0R"  # your Google Drive folder ID

########################################
# Google Drive Auth
########################################
def get_drive_service():
    creds_dict = st.secrets["google_service_account"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials)

def upload_to_gdrive(local_file_path, folder_id=FOLDER_ID):
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
# A small helper to render a color bar
########################################
def toxicity_bar(tox_score: float):
    """
    Returns an HTML snippet that represents a color-coded bar from 0 (green) to 1 (red).
    We'll use HSL to smoothly interpolate color.
    """
    # Clip score to [0,1] in case of float rounding
    tox_score = max(0.0, min(tox_score, 1.0))
    # Convert to a hue from green(120) down to red(0)
    hue = 120 * (1 - tox_score)  # 0 => red(0), 1 => green(120)
    color = f"hsl({hue},100%,50%)"
    bar_width = f"{tox_score * 100:.1f}%"
    
    # Outer container width 200px, or you can do 100% if you prefer
    html_code = f"""
    <div style="background-color:#e0e0e0;width:200px;height:15px;border-radius:5px;overflow:hidden;">
      <div style="background-color:{color};width:{bar_width};height:100%;"></div>
    </div>
    """
    return html_code

########################################
# The Streamlit App
########################################
def main():
    st.title("Data Ingestion & Google Drive Upload")
    st.write("Upload your documents or paste text here to train the language model.")
    
    st.info(f"Note: All ingested files will be uploaded to **{GDRIVE_FOLDER_NAME}**.")

    # -- 1) File Uploader --
    uploaded_files = st.file_uploader(
        "Choose your files (optional)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"]
    )

    if uploaded_files:
        st.subheader("Files ready to be ingested:")
        for uf in uploaded_files:
            st.write(f"- {uf.name}")
    else:
        st.info("No files uploaded yet. You can still paste text below.")

    # -- 2) Text Input Box --
    st.write("---")
    st.subheader("OR Paste Text Directly")
    pasted_text = st.text_area("Paste your text here (optional):", height=150)
    st.caption("If you paste something here, it will also be included in the ingestion.")

    # -- 3) Ingest & Upload Button --
    if st.button("Ingest & Upload to Drive"):
        # 3a) Save uploaded files to disk
        file_paths = []
        with st.spinner("Saving any uploaded files locally..."):
            if uploaded_files:
                os.makedirs("user_uploads", exist_ok=True)
                for uf in uploaded_files:
                    saved_path = os.path.join("user_uploads", uf.name)
                    with open(saved_path, "wb") as f:
                        f.write(uf.getbuffer())
                    file_paths.append(saved_path)

        # 3b) If user pasted text, treat it as a "virtual file"
        # We'll create a temporary .txt file in 'user_uploads'
        temp_txt_path = None
        if pasted_text.strip():
            os.makedirs("user_uploads", exist_ok=True)
            # Generate a unique name for the pasted text
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_txt_path = os.path.join("user_uploads", f"pasted_text_{timestamp}.txt")
            with open(temp_txt_path, "w", encoding="utf-8") as f:
                f.write(pasted_text)
            file_paths.append(temp_txt_path)

        # 3c) If no files *and* no pasted text, warn
        if not file_paths:
            st.warning("No files or pasted text provided. Please upload or paste something.")
            return

        # 3d) Ingest them
        with st.spinner("Ingesting files..."):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"combined_corpus_{timestamp}.txt"
            results = ingest_files(file_paths, output_path=output_path)

        st.subheader("Toxicity Summary (Threshold=0.5)")

        # 3e) Show a small table of results with color-coded bars
        if results:
            for r in results:
                tox = r["toxicity"]
                file_name = r["filename"]
                st.write(f"**{file_name}** => Toxicity: {tox:.2f}")
                # Insert a color-coded bar
                bar_html = toxicity_bar(tox)
                st.markdown(bar_html, unsafe_allow_html=True)

                # If above threshold, highlight
                if tox >= TOXICITY_THRESHOLD:
                    st.warning(f"**High toxicity** detected (≥ 0.5) in {file_name}")

        # 3f) Upload combined corpus to Drive
        st.info(f"Uploading **{output_path}** to Google Drive folder: {GDRIVE_FOLDER_NAME}")
        try:
            upload_to_gdrive(output_path, folder_id=FOLDER_ID)
            st.success("Files have been ingested and uploaded to Google Drive successfully!")
            st.balloons()
        except Exception as e:
            st.error(f"Upload to Drive failed: {e}")
            return

        # 3g) Provide download button
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


