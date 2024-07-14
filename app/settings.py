import streamlit as st

def set_streamlit():
    # Konfiguriere die Streamlit-Seite mit Titel, Icon, Seitenleiste, Layout und Menü
    st.set_page_config(
        page_title="Customer Service Classifier AI",
        page_icon="favicon.ico",
        initial_sidebar_state="expanded",
        layout="centered",
        menu_items={
            'About': "This application is an Experimental Prototype created for my Bachelor Thesis. Created by Kamyab Nazari"
            }
        )

    # Füge ein Logo zur Navigationsleiste hinzu
    st.logo(image="misc/customer_service_classifier_ai_icon.png")

    # Füge eine Überschrift mit Regenbogen-Trenner hinzu
    st.header('Customer Service Classifier AI', divider='rainbow')
