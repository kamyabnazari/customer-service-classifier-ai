import streamlit as st
from services.data_service import list_datasets, load_dataset
from services.openai_service import generate_response
import pandas as pd
import os

# Streamlit app
st.title("Dataset Loader and OpenAI Text Generator")

# List available datasets
data_dir = './data'
datasets = list_datasets(data_dir)

# Select dataset
selected_dataset = st.selectbox("Select a dataset", datasets)

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

# Input field for user to enter their prompt
user_input = st.text_input("Enter your prompt:")

# Button to submit the prompt
if st.button("Generate"):
    if user_input:
        # Generate the response from OpenAI
        output = generate_response(user_input)
        # Display the response
        st.write(f"Response: {output}")
    else:
        st.write("Please enter a prompt.")
