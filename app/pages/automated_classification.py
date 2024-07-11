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
    st.subheader("GPT-3.5 Turbo Zero-Shot")
    st.write("Classify First 10 Samples using GPT-3.5 Turbo Zero-Shot:")
    if st.button("GPT-3.5 Turbo Zero-Shot with Temp 0.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Zero-Shot", 0.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Zero-Shot (Temp 0.0).")
    if st.button("GPT-3.5 Turbo Zero-Shot with Temp 1.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Zero-Shot", 1.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Zero-Shot (Temp 1.0).")
    st.divider()

    st.subheader("GPT-3.5 Turbo Few-Shot")
    st.write("Classify First 10 Samples using GPT-3.5 Turbo Few-Shot:")
    if st.button("GPT-3.5 Turbo Few-Shot with Temp 0.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Few-Shot", 0.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Few-Shot (Temp 0.0).")
    if st.button("GPT-3.5 Turbo Few-Shot with Temp 1.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Few-Shot", 1.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Few-Shot (Temp 1.0).")
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Zero-Shot")
    st.write("Classify First 10 Samples using GPT-3.5 Turbo Fine-Tuned Zero-Shot:")
    if st.button("GPT-3.5 Turbo Fine-Tuned Zero-Shot with Temp 0.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Zero-Shot", 0.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Zero-Shot (Temp 0.0).")
    if st.button("GPT-3.5 Turbo Fine-Tuned Zero-Shot with Temp 1.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Zero-Shot", 1.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Zero-Shot (Temp 1.0).")
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Few-Shot")
    st.write("Classify First 10 Samples using GPT-3.5 Turbo Fine-Tuned Few-Shot:")
    if st.button("GPT-3.5 Turbo Fine-Tuned Few-Shot with Temp 0.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Few-Shot", 0.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Few-Shot (Temp 0.0).")
    if st.button("GPT-3.5 Turbo Fine-Tuned Few-Shot with Temp 1.0", use_container_width=True):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Few-Shot", 1.0, classification_method="automated")
        st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Few-Shot (Temp 1.0).")