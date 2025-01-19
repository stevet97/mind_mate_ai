import streamlit as st
import os
import datetime
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Import the ingestion function from data_ingestion/data_ingestion.py
from data_ingestion.data_ingestion import ingest_files

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

########################################
# Constants / Config
########################################
TOXICITY_THRESHOLD = 0.5
SKIP_TOXIC = True  # Exclude items with toxicity >= 0.5
GDRIVE_FOLDER_NAME = "My Drive / NOVA_project_data"
FOLDER_ID = "1fpMg9W19LF6YDJVjgUq2aylUqPfF1M0R"

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
    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(local_file_path, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()
    return uploaded_file.get("id")

########################################
# Toxicity Bar Helper
########################################
def toxicity_bar(tox_score: float):
    tox_score = max(0.0, min(tox_score, 1.0))
    hue = 120 * (1 - tox_score)
    color = f"hsl({hue},100%,50%)"
    bar_width = f"{tox_score * 100:.1f}%"
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
    st.title("Data Ingestion & Drive Upload (Skip Toxic ≥ 0.5)")
    st.info(f"Files above toxicity {TOXICITY_THRESHOLD} are excluded from the final corpus.")

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
        st.info("No files uploaded yet. You can also paste text below.")

    st.write("---")
    st.subheader("OR Paste Text Directly")
    pasted_text = st.text_area("Paste text here (optional):", height=150)
    st.caption("It will be treated as one additional .txt file in the pipeline.")

    if st.button("Ingest & Upload to Drive"):
        file_paths = []
        with st.spinner("Saving uploads..."):
            if uploaded_files:
                os.makedirs("user_uploads", exist_ok=True)
                for uf in uploaded_files:
                    saved_path = os.path.join("user_uploads", uf.name)
                    with open(saved_path, "wb") as f:
                        f.write(uf.getbuffer())
                    file_paths.append(saved_path)

        if pasted_text.strip():
            os.makedirs("user_uploads", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_txt_path = os.path.join("user_uploads", f"pasted_text_{timestamp}.txt")
            with open(temp_txt_path, "w", encoding="utf-8") as f:
                f.write(pasted_text)
            file_paths.append(temp_txt_path)

        if not file_paths:
            st.warning("No files or pasted text provided.")
            return

        with st.spinner("Ingesting files..."):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"combined_corpus_{timestamp}.txt"
            results = ingest_files(
                file_paths,
                output_path=output_path,
                skip_toxic=SKIP_TOXIC,
                toxicity_threshold=TOXICITY_THRESHOLD
            )

        st.subheader(f"Toxicity Summary (≥ {TOXICITY_THRESHOLD} excluded)")

        if results:
            included = []
            excluded = []
            for r in results:
                if r["toxicity"] >= TOXICITY_THRESHOLD:
                    excluded.append(r)
                else:
                    included.append(r)

            st.write("**Included:**")
            if included:
                for r in included:
                    st.write(f"{r['filename']} => {r['toxicity']:.2f}")
                    st.markdown(toxicity_bar(r['toxicity']), unsafe_allow_html=True)
            else:
                st.info("All items were excluded (toxicity ≥ 0.5).")

            st.write("---")

            st.write("**Excluded:**")
            if excluded:
                for r in excluded:
                    st.write(f"{r['filename']} => {r['toxicity']:.2f}")
                    st.markdown(toxicity_bar(r['toxicity']), unsafe_allow_html=True)
                st.warning(f"{len(excluded)} item(s) excluded.")
            else:
                st.info("No items were above 0.5 toxicity.")

        st.info(f"Uploading {output_path} to {GDRIVE_FOLDER_NAME} in Drive...")
        try:
            upload_to_gdrive(output_path, folder_id=FOLDER_ID)
            st.success("Ingestion & Upload successful!")
            st.balloons()
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return

        st.write("---")
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download final combined_corpus",
                data=f,
                file_name="combined_corpus.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()

