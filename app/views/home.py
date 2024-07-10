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

# Button to load the selected dataset
if st.button("Load Dataset"):
    try:
        if selected_dataset:
            # Convert back to lowercase to match the folder names
            selected_dataset = selected_dataset.lower()
            global_state.datasets = load_datasets(selected_dataset, data_dir)
            st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

# Display a sample of the loaded data
if "test" in global_state.datasets:
    st.write("First element from 'test.csv':")
    st.write(global_state.datasets["test"].iloc[0, 0])
    st.write("First row from 'test.csv':")
    st.write(global_state.datasets["test"].iloc[0])

if "fine_tuning" in global_state.datasets:
    st.write("First element from 'fine_tuning.csv':")
    st.write(global_state.datasets["fine_tuning"].iloc[0, 0])
    st.write("First row from 'fine_tuning.csv':")
    st.write(global_state.datasets["fine_tuning"].iloc[0])