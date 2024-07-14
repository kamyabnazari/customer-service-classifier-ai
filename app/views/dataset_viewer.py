import streamlit as st
from state import global_state

# Setze den Titel der Seite
st.title("Dataset Viewer")

# Überprüfe, ob ein Datensatz geladen wurde, und ermögliche die Rückkehr zur Startseite, falls nicht
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("views/home.py")
else:
    # Anzeige der Kategorien-Daten, falls vorhanden
    if "categories" in global_state.datasets:
        st.write("Categories")
        st.dataframe(global_state.datasets["categories"], use_container_width=True)
        st.divider()

    # Anzeige der Testdaten, falls vorhanden
    if "test" in global_state.datasets:
        st.write("Test Data")
        st.dataframe(global_state.datasets["test"], use_container_width=True)
        st.divider()

    # Anzeige der Daten für das Feintuning, falls vorhanden
    if "fine_tuning" in global_state.datasets:
        st.write("Fine Tuning Data")
        st.dataframe(global_state.datasets["fine_tuning"], use_container_width=True)
