import streamlit as st
import settings

# Streamlit app
settings.set_streamlit()

# Define the pages
pages = [
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/dataset_viewer.py", title="Dataset Viewer"),
    st.Page("pages/experiment.py", title="Experiment")
]

# Display the selected page
pg = st.navigation(pages)
pg.run()