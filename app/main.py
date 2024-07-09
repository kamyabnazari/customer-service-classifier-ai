import streamlit as st
from config import settings
from dependencies import get_database

database = get_database()

# Streamlit app
st.title("OpenAI Question Answering App")
