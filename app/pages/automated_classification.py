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
    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Zero-Shot (Temp 0.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo", "Zero-Shot", 0.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Zero-Shot (Temp 0.0).")
    with col4:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Zero-Shot (Temp 1.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo", "Zero-Shot", 1.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Zero-Shot (Temp 1.0).")
    st.divider()

    st.subheader("GPT-3.5 Turbo Few-Shot")
    col5, col6 = st.columns([1, 1])
    with col5:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Few-Shot (Temp 0.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo", "Few-Shot", 0.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Few-Shot (Temp 0.0).")
    with col6:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Few-Shot (Temp 1.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo", "Few-Shot", 1.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Few-Shot (Temp 1.0).")
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Zero-Shot")
    col7, col8 = st.columns([1, 1])
    with col7:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Fine-Tuned Zero-Shot (Temp 0.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Zero-Shot", 0.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Zero-Shot (Temp 0.0).")
    with col8:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Fine-Tuned Zero-Shot (Temp 1.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Zero-Shot", 1.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Zero-Shot (Temp 1.0).")
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Few-Shot")
    col9, col10 = st.columns([1, 1])
    with col9:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Fine-Tuned Few-Shot (Temp 0.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Few-Shot", 0.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Few-Shot (Temp 0.0).")
    with col10:
        if st.button("Classify First 10 Samples - GPT-3.5 Turbo Fine-Tuned Few-Shot (Temp 1.0)", use_container_width=True):
            classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Few-Shot", 1.0)
            st.success("First 10 samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Few-Shot (Temp 1.0).")