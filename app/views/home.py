import streamlit as st
from services.openai_service import generate_response

# Set page title
st.title("Home")
st.sidebar.title("Options")

left, right = st.columns(2, vertical_alignment="bottom")

# Input field for user to enter their prompt
user_input = left.text_input("Enter your prompt:")
# Button to submit the prompt
if right.button("Generate"):
    if user_input:
        # Generate the response from OpenAI
        output = generate_response(user_input)
        # Display the response
        st.write(f"Response: {output}")
    else:
        st.write("Please enter a prompt.")
