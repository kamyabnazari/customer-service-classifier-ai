import streamlit as st
from services.openai_service import (
    classify_with_gpt_3_5_turbo,
    classify_with_gpt_4,
    classify_with_gpt_4_turbo,
    classify_with_gpt_4o
    )

st.title("Experiment")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("pages/home.py")
else:
    # Select model
    model_option = st.selectbox("Select a model", ["GPT-3.5 Turbo", "GPT-4", "GPT-4 Turbo", "GPT-4o"])

    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

    with col1:
        # Input field for user to enter their prompt
        user_input = st.text_input("Enter text to classify:")

    classification = ""

    with col2:
    # Button to submit the prompt
        if st.button("Classify"):
            if user_input:
                # Generate the response from OpenAI
                if model_option == "GPT-3.5 Turbo":
                    classification = classify_with_gpt_3_5_turbo(user_input)
                elif model_option == "GPT-4":
                    classification = classify_with_gpt_4(user_input)
                elif model_option == "GPT-4 Turbo":
                    classification = classify_with_gpt_4_turbo(user_input)
                elif model_option == "GPT-4o":
                    classification = classify_with_gpt_4o(user_input)
            else:
                st.write("Please enter a text to classify.")

    st.divider()

    # Display the response
    if classification != "":
        st.write(f"Classification: {classification}")

