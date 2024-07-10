import streamlit as st
from services.data_service import list_datasets, load_datasets
from state import global_state

# Set page title
st.title("Home")

# Initialize session state for dataset loaded state
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

# List available datasets
data_dir = './data'
datasets = list_datasets(data_dir)

# Capitalize the first letter of each dataset name
datasets_capitalized = [dataset.capitalize() for dataset in datasets]

col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

with col1:
    # Select dataset
    selected_dataset = st.selectbox("Select a dataset", datasets_capitalized, disabled=st.session_state.dataset_loaded)

with col2:
    # Conditionally render buttons
    if not st.session_state.dataset_loaded:
        # Load dataset button
        if st.button("Load Dataset", use_container_width=True):
            try:
                if selected_dataset:
                    # Convert back to lowercase to match the folder names
                    selected_dataset = selected_dataset.lower()
                    global_state.datasets = load_datasets(selected_dataset, data_dir)
                    st.session_state.dataset_loaded = True
                    st.success("Dataset loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
    else:
        # Delete dataset button
        if st.button("Delete Dataset", type="primary", use_container_width=True):
            global_state.datasets.clear()
            st.session_state.dataset_loaded = False
            st.success("Dataset deleted successfully!")
            st.rerun()

st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

# Display a sample of the loaded data
if global_state.datasets:
    with col1:
        if "test" in global_state.datasets:
            st.metric(label="Total Test Elements", value=len(global_state.datasets["test"]))

    with col2:
        if "fine_tuning" in global_state.datasets:
            st.metric(label="Total Fine Tuning Elements", value=len(global_state.datasets["fine_tuning"]))

    with col3:
        if "categories" in global_state.datasets:
            st.metric(label="Total Category Elements", value=len(global_state.datasets["categories"]))