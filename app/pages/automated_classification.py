import streamlit as st
from state import global_state
from services.experiment_service import classify_test_samples

st.title("Automated Classification")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("pages/home.py")
else:
    st.write("Classify the first 10 samples using the following methods:")

    st.subheader("GPT-3.5 Turbo Zero-Shot")
    if st.button("Classify First 10 Samples - GPT-3.5 Turbo Zero-Shot", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Zero-Shot")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Zero-Shot.")
    st.divider()

    st.subheader("GPT-3.5 Turbo Few-Shot")
    if st.button("Classify First 10 Samples - GPT-3.5 Turbo Few-Shot", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Few-Shot")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Few-Shot.")
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Zero-Shot")
    if st.button("Classify First 10 Samples - GPT-3.5 Turbo Fine-Tuned Zero-Shot", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Zero-Shot")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Zero-Shot.")
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Few-Shot")
    if st.button("Classify First 10 Samples - GPT-3.5 Turbo Fine-Tuned Few-Shot", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Few-Shot")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Few-Shot.")
