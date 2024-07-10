import streamlit as st

def set_streamlit():
    # Set app Config and Icon
    st.set_page_config(
        page_title="Customer Service Classifier AI",
        page_icon="favicon.ico",
        initial_sidebar_state="expanded",
        layout="centered",
        menu_items={
            'About': "This application is an Experimental Prototype created for my Bachelor Thesis. Created by Kamyab Nazari"
            }
        )

    # Set Navigation Bar Logo
    st.logo(image="misc/customer_service_classifier_ai_icon.png")

    # Set Application Header 
    st.header('Customer Service Classifier AI', divider='rainbow')