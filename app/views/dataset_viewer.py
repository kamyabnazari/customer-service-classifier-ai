import streamlit as st
from services.data_service import list_datasets, load_dataset
import pandas as pd
import os

# Set page title
st.title("Dataset Viewer")
st.sidebar.title("Dataset Options")

# List available datasets
data_dir = './data'
datasets = list_datasets(data_dir)

left, right = st.columns(2, vertical_alignment="top")

# Select dataset
with left:
    selected_dataset = st.selectbox("Select a dataset", datasets)

with right:
    # Load and display selected dataset
    if selected_dataset:
        dataset_path = os.path.join(data_dir, selected_dataset)
        dataset = load_dataset(dataset_path)
        
        st.write(f"Dataset: {selected_dataset}")
        for file_name, data in dataset.items():
            st.write(f"File: {file_name}")
            if isinstance(data, pd.DataFrame):
                st.dataframe(data)
            else:
                st.json(data)