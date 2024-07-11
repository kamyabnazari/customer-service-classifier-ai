from openai import OpenAI
from config import OPENAI_API_KEY
import streamlit as st
from services.data_service import csv_to_jsonl
import os

# Set up the OpenAI API client
openai = OpenAI(api_key=OPENAI_API_KEY)

st.title("Utility Tools")


# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("pages/home.py")
else:
    # Laden der CSV in JSONL-Konvertierungslogik
    st.header("Preparing Fine Tuning")

    # Pfade für CSV und JSONL Dateien
    csv_file_path = './data/banking/fine_tuning.csv'
    jsonl_file_path = './data/banking_generated/fine_tuning.jsonl'

    # Überprüfen, ob die JSONL-Datei bereits existiert
    jsonl_file_exists = os.path.isfile(jsonl_file_path)

    if st.button("Convert CSV to JSONL", disabled=jsonl_file_exists, use_container_width=True):
        csv_to_jsonl(csv_file_path, jsonl_file_path, "You are a Customer Service Inquiry classifier.")
        st.success(f"CSV file successfully converted to JSONL and saved to {jsonl_file_path}.")
        st.rerun()

    # Nachricht anzeigen, wenn die JSONL-Datei bereits existiert
    if jsonl_file_exists:
        st.info(f"JSONL file already exists at {jsonl_file_path}.")

    st.header("Delete all Files from OpenAI")

    if st.button("Delete All Files", use_container_width=True):
        try:
            files = openai.files.list().data
            for file in files:
                openai.files.delete(file.id)
            st.success("All files deleted from OpenAI!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to delete files from OpenAI: {e}")

    st.divider()