from openai import OpenAI
from config import OPENAI_API_KEY
import streamlit as st
from services.data_service import csv_to_jsonl, list_jsonl_files
from services.openai_service import list_fine_tune_jobs, upload_file_to_openai, fine_tune_model, get_fine_tune_status
import os

openai = OpenAI(api_key=OPENAI_API_KEY)

st.title("Utility Tools")

if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("views/home.py")
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

    # Example Streamlit app
    st.header("Fine-Tuning Model")

    # Verzeichnis für JSONL-Dateien
    jsonl_directory = './data/banking_generated'

    # Liste der JSONL-Dateien im Verzeichnis
    jsonl_files = list_jsonl_files(jsonl_directory)

    # Dropdown-Auswahl für JSONL-Datei
    selected_jsonl_file = st.selectbox("Select a JSONL file", jsonl_files)

    if selected_jsonl_file:
        jsonl_file_path = os.path.join(jsonl_directory, selected_jsonl_file)

        if st.button("Upload JSONL File", use_container_width=True):
            try:
                upload_response = upload_file_to_openai(jsonl_file_path)
                training_file_id = upload_response.id
                st.session_state.training_file_id = training_file_id

                st.success(f"Uploaded Fine Tuning File ID: {training_file_id}")
            except Exception as e:
                st.error(f"Failed to upload file: {e}")

    if "training_file_id" in st.session_state:
        training_file_id = st.session_state.training_file_id

        if st.button("Create Fine-Tuning Job", use_container_width=True):
            try:
                fine_tuning_job_response = fine_tune_model(training_file_id)
                fine_tuning_job_id = fine_tuning_job_response.id

                st.session_state.fine_tuning_job_id = fine_tuning_job_id

                st.success(f"Fine-tuning started with ID: {fine_tuning_job_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start fine-tuning: {e}")

    st.divider()

    st.header("Check Fine-Tune Job Status")

    fine_tune_jobs = list_fine_tune_jobs()
    fine_tune_job_options = {f"{job.id} ({job.status})": job.id for job in fine_tune_jobs}

    selected_fine_tune_job = st.selectbox("Select a Fine-Tune Job", list(fine_tune_job_options.keys()))

    if selected_fine_tune_job:
        fine_tuning_job_id = fine_tune_job_options[selected_fine_tune_job]
        
        if st.button("Check Selected Fine-Tune Status", use_container_width=True):
            try:
                status_response = get_fine_tune_status(fine_tuning_job_id)
                status = status_response.status
                
                if status in ["validating_files", "queued", "running"]:
                    st.info(f"Fine-tune status: {status}")
                elif status in ["failed", "cancelled"]:
                    st.error(f"Fine-tune status: {status}")
                elif status == "succeeded":
                    st.success(f"Fine-tune status: {status}")
            except Exception as e:
                st.error(f"Failed to retrieve fine-tune status: {e}")