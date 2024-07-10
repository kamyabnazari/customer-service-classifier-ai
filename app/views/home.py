import pandas as pd
import streamlit as st
from services.data_service import list_datasets, load_datasets
from state import global_state

# Set page title
st.title("Home")
st.sidebar.title("Options")

# List available datasets
data_dir = './data'
datasets = list_datasets(data_dir)

# Capitalize the first letter of each dataset name
datasets_capitalized = [dataset.capitalize() for dataset in datasets]

# Select dataset
selected_dataset = st.selectbox("Select a dataset", datasets_capitalized)

# Initialize session state for load button disabled state
if "load_disabled" not in st.session_state:
    st.session_state.load_disabled = False

# Layout for buttons
col1, col2 = st.columns([1, 4])

# Load dataset button
with col1:
    if st.button("Load Dataset", disabled=st.session_state.load_disabled):
        try:
            if selected_dataset:
                # Convert back to lowercase to match the folder names
                selected_dataset = selected_dataset.lower()
                global_state.datasets = load_datasets(selected_dataset, data_dir)
                st.session_state.load_disabled = True
                st.success("Dataset loaded successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

# Delete dataset button
with col2:
    if st.button("Delete Dataset", type="primary", disabled=not st.session_state.load_disabled):
        global_state.datasets.clear()
        st.session_state.load_disabled = False
        st.success("Dataset deleted successfully!")
        st.rerun()

# Display a sample of the loaded data
if global_state.datasets:
    data_stats = []
    if "test" in global_state.datasets:
        data_stats.append({
            "File": "test.csv",
            "Total Elements": len(global_state.datasets["test"])
        })

    if "fine_tuning" in global_state.datasets:
        data_stats.append({
            "File": "fine_tuning.csv",
            "Total Elements": len(global_state.datasets["fine_tuning"])
        })

    if "categories" in global_state.datasets:
        data_stats.append({
            "File": "categories.json",
            "Total Elements": len(global_state.datasets["categories"])
        })

    # Display data statistics
    if data_stats:
        st.write("Data Statistics")
        stats_df = pd.DataFrame(data_stats)
        st.dataframe(stats_df)