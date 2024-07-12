import streamlit as st
from state import global_state
from services.openai_service import (
    classify_with_gpt_3_5_turbo_zero_shot,
    classify_with_gpt_3_5_turbo_few_shot,
    classify_with_gpt_3_5_turbo_fine_zero_shot,
    classify_with_gpt_3_5_turbo_fine_few_shot,
    classify_with_gpt_3_5_turbo_fine_no_prompting
)

st.title("Manual Classification")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("views/home.py")
else:
    # Select model and method for manual classification
    model_option = st.selectbox("Select a model", ["GPT-3.5 Turbo", "GPT-3.5 Turbo Fine-Tuned"])
    
    # Conditionally enable "No Prompting" method
    if model_option == "GPT-3.5 Turbo Fine-Tuned":
        method_option = st.selectbox("Select a method", ["Zero-Shot", "Few-Shot", "No Prompting"])
    else:
        method_option = st.selectbox("Select a method", ["Zero-Shot", "Few-Shot"])
    
    temperature_option = st.selectbox("Select Temperature", [0.0, 1.0])

    # Input field for user to enter their prompt
    user_input = st.text_input("Enter text to classify:")
    true_category = st.text_input("Enter true category (optional):")

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
                    classification = classify_with_gpt_3_5_turbo_zero_shot(user_input, categories, temperature_option, classification_method="manual", true_category=true_category)
                elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                    classification = classify_with_gpt_3_5_turbo_fine_zero_shot(user_input, categories, temperature_option, classification_method="manual", true_category=true_category)
            elif method_option == "Few-Shot":
                if model_option == "GPT-3.5 Turbo":
                    classification = classify_with_gpt_3_5_turbo_few_shot(user_input, categories, temperature_option, classification_method="manual", true_category=true_category)
                elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                    classification = classify_with_gpt_3_5_turbo_fine_few_shot(user_input, categories, temperature_option, classification_method="manual", true_category=true_category)
            elif method_option == "No Prompting":
                if model_option == "GPT-3.5 Turbo Fine-Tuned":
                    classification = classify_with_gpt_3_5_turbo_fine_no_prompting(user_input, temperature_option, classification_method="manual", true_category=true_category)

        # Display the response
        if classification != "":
            st.info(f"Classification: {classification}")
        else:
            st.write("Please enter a text to classify.")

    st.divider()

    if "test" in global_state.datasets:
        st.write("Test Data")
        st.dataframe(global_state.datasets["test"], use_container_width=True)
