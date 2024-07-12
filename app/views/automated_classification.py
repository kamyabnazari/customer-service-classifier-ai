import streamlit as st
from state import global_state
from services.experiment_service import classify_test_samples
from services.data_service import check_results_exist

st.title("Automated Classification")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("views/home.py")
else:
    st.subheader("GPT-3.5 Turbo Zero-Shot")
    st.write("Classify Samples using GPT-3.5 Turbo Zero-Shot:")
    results_exist = check_results_exist("gpt-3.5-turbo-0125", "zero_shot", 0.0, "automated")
    if st.button("GPT-3.5 Turbo Zero-Shot with Temp 0.0", use_container_width=True, disabled=results_exist):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Zero-Shot", 0.0, classification_method="automated")
        st.success("All samples classified and results saved using GPT-3.5 Turbo Zero-Shot (Temp 0.0).")
        st.rerun()
    st.divider()

    st.subheader("GPT-3.5 Turbo Few-Shot")
    st.write("Classify Samples using GPT-3.5 Turbo Few-Shot:")
    results_exist = check_results_exist("gpt-3.5-turbo-0125", "few_shot", 0.0, "automated")
    if st.button("GPT-3.5 Turbo Few-Shot with Temp 0.0", use_container_width=True, disabled=results_exist):
        classify_test_samples(global_state, "GPT-3.5 Turbo", "Few-Shot", 0.0, classification_method="automated")
        st.success("All samples classified and results saved using GPT-3.5 Turbo Few-Shot (Temp 0.0).")
        st.rerun()
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Zero-Shot")
    st.write("Classify Samples using GPT-3.5 Turbo Fine-Tuned Zero-Shot:")
    results_exist = check_results_exist("ft:gpt-3.5-turbo-0125:personal:classifier-model:9k0H1hdV", "fine_zero_shot", 0.0, "automated")
    if st.button("GPT-3.5 Turbo Fine-Tuned Zero-Shot with Temp 0.0", use_container_width=True, disabled=results_exist):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Zero-Shot", 0.0, classification_method="automated")
        st.success("All samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Zero-Shot (Temp 0.0).")
        st.rerun()
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned Few-Shot")
    st.write("Classify Samples using GPT-3.5 Turbo Fine-Tuned Few-Shot:")
    results_exist = check_results_exist("ft:gpt-3.5-turbo-0125:personal:classifier-model:9k0H1hdV", "fine_few_shot", 0.0, "automated")
    if st.button("GPT-3.5 Turbo Fine-Tuned Few-Shot with Temp 0.0", use_container_width=True, disabled=results_exist):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "Few-Shot", 0.0, classification_method="automated")
        st.success("All samples classified and results saved using GPT-3.5 Turbo Fine-Tuned Few-Shot (Temp 0.0).")
        st.rerun()
    st.divider()

    st.subheader("GPT-3.5 Turbo Fine-Tuned No Prompting")
    st.write("Classify Samples using GPT-3.5 Turbo Fine-Tuned No Prompting:")
    results_exist = check_results_exist("ft:gpt-3.5-turbo-0125:personal:classifier-model:9k0H1hdV", "fine_no_prompting", 0.0, "automated")
    if st.button("GPT-3.5 Turbo Fine-Tuned No Prompting with Temp 0.0", use_container_width=True, disabled=results_exist):
        classify_test_samples(global_state, "GPT-3.5 Turbo Fine-Tuned", "No Prompting", 0.0, classification_method="automated")
        st.success("All samples classified and results saved using GPT-3.5 Turbo Fine-Tuned No Prompting (Temp 0.0).")
        st.rerun()