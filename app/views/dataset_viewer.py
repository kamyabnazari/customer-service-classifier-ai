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

# Capitalize the first letter of each dataset name
datasets_capitalized = [dataset.capitalize() for dataset in datasets]

# Select dataset
selected_dataset = st.selectbox("Select a dataset", datasets_capitalized)

# Initialize variables for the datasets
test_data = None
fine_tuning_data = None
categories_data = None

# Load the selected dataset
if selected_dataset:
    # Convert back to lowercase to match the folder names
    selected_dataset = selected_dataset.lower()
    dataset_path = os.path.join(data_dir, selected_dataset)
    dataset = load_dataset(dataset_path)
    
    # Assign the loaded data to the respective variables
    for file_name, data in dataset.items():
        if file_name == 'test.csv':
            test_data = pd.DataFrame(data)
        elif file_name == 'fine_tuning.csv':
            fine_tuning_data = pd.DataFrame(data)
        elif file_name == 'categories.json':
            categories_data = data

# Display the datasets side by side
left, right = st.columns(2)

with left:
    if test_data is not None:
        st.write("Test Data")
        st.dataframe(test_data)

with right:
    if fine_tuning_data is not None:
        st.write("Fine Tuning Data")
        st.dataframe(fine_tuning_data)

# Display the categories list full width
if categories_data is not None:
    st.write("Categories")
    st.json(categories_data)
