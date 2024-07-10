import streamlit as st
import settings

# Streamlit app
settings.set_streamlit()

# Define the pages
pages = {
    "Home": [st.Page("views/home.py", title="Customer Service Classifier AI")],
    "Dataset Viewer": [st.Page("views/dataset_viewer.py", title="Dataset Viewer")]
}

# Display the selected page
pg = st.navigation(pages)
pg.run()