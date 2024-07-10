import streamlit as st
from state import global_state

# Set page title
st.title("Dataset Viewer")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("pages/home.py")
else:
    if "test" in global_state.datasets:
        st.write("Test Data")
        st.dataframe(global_state.datasets["test"], use_container_width=True)

    if "fine_tuning" in global_state.datasets:
        st.write("Fine Tuning Data")
        st.dataframe(global_state.datasets["fine_tuning"], use_container_width=True)

    # Display the categories list full width
    if "categories" in global_state.datasets:
        st.write("Categories")
        st.json(global_state.datasets["categories"])