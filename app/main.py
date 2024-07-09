import streamlit as st
from dependencies import get_database
from config import settings

database = get_database()

# Streamlit app
st.title("OpenAI Question Answering App")
