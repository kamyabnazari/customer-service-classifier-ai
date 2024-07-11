import streamlit as st
import settings

# Streamlit app
settings.set_streamlit()

# Define the pages
pages = [
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/dataset_viewer.py", title="Dataset Viewer"),
    st.Page("pages/manual_classification.py", title="Manual Classification"),
    st.Page("pages/automated_classification.py", title="Automated Classification"),
    st.Page("pages/utility_tools.py", title="Utility Tools")
]

# Display the selected page
pg = st.navigation(pages)
pg.run()