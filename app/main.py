import streamlit as st
import settings

# Streamlit app
settings.set_streamlit()

# Define the pages
pages = [
    st.Page("views/home.py", title="Home"),
    st.Page("views/dataset_viewer.py", title="Dataset Viewer"),
    st.Page("views/experiment.py", title="Experiment")
]

# Display the selected page
pg = st.navigation(pages)
pg.run()