import streamlit as st
from state import global_state
from services.openai_service import (
    classify_with_gpt_3_5_turbo_zero_shot,
    classify_with_gpt_3_5_turbo_few_shot,
    classify_with_gpt_3_5_turbo_fine_zero_shot,
    classify_with_gpt_3_5_turbo_fine_few_shot
)

st.title("Manual Classification")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("pages/home.py")
else:
    # Select model and method for manual classification
    model_option = st.selectbox("Select a model", ["GPT-3.5 Turbo", "GPT-3.5 Turbo Fine-Tuned"])
    method_option = st.selectbox("Select a method", ["Zero-Shot", "Few-Shot"])

    # Input field for user to enter their prompt
    user_input = st.text_input("Enter text to classify:")

    classification = ""

    # Extract categories from the dataset
    if "categories" in global_state.datasets:
        categories = global_state.datasets["categories"].iloc[:, 0].tolist()
    else:
        st.error("Categories not found in the dataset.")
        st.stop()

    # Button to submit the prompt
    if st.button("Classify", use_container_width=True):
        if user_input:
            # Generate the response from OpenAI
            if method_option == "Zero-Shot":
                if model_option == "GPT-3.5 Turbo":
                    classification = classify_with_gpt_3_5_turbo_zero_shot(user_input, categories)
                elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                    classification = classify_with_gpt_3_5_turbo_fine_zero_shot(user_input, categories)
            elif method_option == "Few-Shot":
                if model_option == "GPT-3.5 Turbo":
                    classification = classify_with_gpt_3_5_turbo_few_shot(user_input, categories)
                elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                    classification = classify_with_gpt_3_5_turbo_fine_few_shot(user_input, categories)

        else:
            st.write("Please enter a text to classify.")

    st.divider()

    # Display the response
    if classification != "":
        st.write(f"Classification: {classification}")