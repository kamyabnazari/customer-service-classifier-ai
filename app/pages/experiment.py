import streamlit as st
from services.openai_service import generate_response
from state import global_state

st.title("Experiment Page")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("pages/home.py")
else:
    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

    with col1:
        # Input field for user to enter their prompt
        user_input = st.text_input("Enter your prompt:")

    with col2:
        # Button to submit the prompt
        if st.button("Generate", use_container_width=True):
            if user_input:
                # Generate the response from OpenAI
                output = generate_response(user_input)
                # Display the response
                st.write(f"Response: {output}")
            else:
                st.write("Please enter a prompt.")
    
    st.divider()