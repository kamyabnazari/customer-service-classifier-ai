import streamlit as st
from state import global_state

# Set page title
st.title("Dataset Viewer")
st.sidebar.title("Dataset Options")

if "test" in global_state.datasets:
    st.write("Test Data")
    st.dataframe(global_state.datasets["test"])

if "fine_tuning" in global_state.datasets:
    st.write("Fine Tuning Data")
    st.dataframe(global_state.datasets["fine_tuning"])

# Display the categories list full width
if "categories" in global_state.datasets:
    st.write("Categories")
    st.json(global_state.datasets["categories"])