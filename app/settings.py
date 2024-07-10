import streamlit as st

def set_streamlit():
    # Set app Config and Icon
    st.set_page_config(page_title="Customer Service Classifier AI", page_icon="🤖")

    # Set Navigation Bar Logo
    st.logo(image="misc/customer_service_classifier_ai_icon.png")
    st.sidebar.title("Navigation")
