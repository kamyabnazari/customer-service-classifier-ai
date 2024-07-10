import streamlit as st
from services.openai_service import generate_response

# Streamlit app
st.title("OpenAI Text Generator")

# Input field for user to enter their prompt
user_input = st.text_input("Enter your prompt:")

# Button to submit the prompt
if st.button("Generate"):
    if user_input:
        # Generate the response from OpenAI
        output = generate_response(user_input)
        # Display the response
        st.write(f"Response: {output}")
    else:
        st.write("Please enter a prompt.")