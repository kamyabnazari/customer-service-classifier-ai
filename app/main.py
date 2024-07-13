import streamlit as st
import settings

# Streamlit app
settings.set_streamlit()

# Define the pages
pages = [
    st.Page("views/home.py", title="Home"),
    st.Page("views/dataset_viewer.py", title="Dataset Viewer"),
    st.Page("views/manual_classification.py", title="Manual Classification"),
    st.Page("views/automated_classification.py", title="Automated Classification"),
    st.Page("views/evaluation.py", title="Evaluation"),
    st.Page("views/evaluation_all.py", title="Evaluation All"),
    st.Page("views/utility_tools.py", title="Utility Tools"),
]

# Display the selected page
pg = st.navigation(pages)
pg.run()