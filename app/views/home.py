from openai import OpenAI
from config import OPENAI_API_KEY
import streamlit as st
from state import global_state
from services.data_service import list_datasets, load_datasets

# Initialisiere den OpenAI API-Client mit dem bereitgestellten API-Schlüssel
openai = OpenAI(api_key=OPENAI_API_KEY)

st.title("Home")

# Initialisiere den Session-State für den Ladestatus des Datensatzes, falls noch nicht geschehen
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

data_dir = './data'
datasets = list_datasets(data_dir)

# Großschreibung des ersten Buchstabens jedes Datensatznamens für Anzeigezwecke
datasets_capitalized = [dataset.capitalize() for dataset in datasets]

st.header("Importing Datasets")

col1, col2 = st.columns([3, 1], vertical_alignment="bottom")

with col1:
    # Dropdown zur Auswahl eines Datensatzes, deaktiviert, wenn bereits ein Datensatz geladen ist
    selected_dataset = st.selectbox("Select a dataset", datasets_capitalized, disabled=st.session_state.dataset_loaded)

with col2:
    # Bedingte Darstellung der Laden- und Löschen-Schaltflächen basierend auf dem Ladestatus des Datensatzes
    if not st.session_state.dataset_loaded:
        if st.button("Load Dataset", use_container_width=True):
            try:
                if selected_dataset:
                    selected_dataset = selected_dataset.lower()
                    global_state.datasets = load_datasets(selected_dataset, data_dir)
                    st.session_state.dataset_loaded = True
                    st.success("Dataset loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
    else:
        if st.button("Delete Dataset", type="primary", use_container_width=True):
            global_state.datasets.clear()
            st.session_state.dataset_loaded = False
            st.success("Dataset deleted successfully!")
            st.rerun()

# Zeige Details des Datensatzes an, wenn ein Datensatz geladen ist
if global_state.datasets:
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if "test" in global_state.datasets:
            st.metric(label="Total Test Elements", value=len(global_state.datasets["test"]))

    with col2:
        if "fine_tuning" in global_state.datasets:
            st.metric(label="Total Fine Tuning Elements", value=len(global_state.datasets["fine_tuning"]))

    with col3:
        if "categories" in global_state.datasets:
            st.metric(label="Total Category Elements", value=len(global_state.datasets["categories"]))