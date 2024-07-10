import streamlit as st
from config import openai

# Streamlit app
st.title("OpenAI Text Generator")

# Input field for user to enter their prompt
user_input = st.text_input("Enter your prompt:")

# Function to generate response from OpenAI
def generate_response(prompt):
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Button to submit the prompt
if st.button("Generate"):
    if user_input:
        # Generate the response from OpenAI
        output = generate_response(user_input)
        # Display the response
        st.write(f"Response: {output}")
    else:
        st.write("Please enter a prompt.")