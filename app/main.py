import streamlit as st
import settings

# Initialisiere Streamlit-Konfigurationen aus dem settings-Modul
settings.set_streamlit()

# Definiere die Seiten der Anwendung
pages = [
    st.Page("views/home.py", title="Home"),
    st.Page("views/dataset_viewer.py", title="Dataset Viewer"),
    st.Page("views/manual_classification.py", title="Manual Classification"),
    st.Page("views/automated_classification.py", title="Automated Classification"),
    st.Page("views/evaluation.py", title="Evaluation"),
    st.Page("views/evaluation_all.py", title="Evaluation All"),
    st.Page("views/utility_tools.py", title="Utility Tools"),
]

# Zeige die ausgewählte Seite an
pg = st.navigation(pages)
pg.run()
