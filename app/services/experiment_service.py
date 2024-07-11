import streamlit as st
import pandas as pd
from services.openai_service import (
    classify_with_gpt_3_5_turbo_zero_shot,
    classify_with_gpt_3_5_turbo_few_shot,
    classify_with_gpt_4o_zero_shot,
    classify_with_gpt_4o_few_shot,
    classify_with_gpt_3_5_turbo_fine_zero_shot,
    classify_with_gpt_3_5_turbo_fine_few_shot
)
from services.logging_service import write_results_to_csv

def classify_test_samples(global_state, model_option, method_option):
    if "test" in global_state.datasets:
        test_data = global_state.datasets["test"]
        
        # Überprüfen, ob die erforderlichen Spalten vorhanden sind
        if 'text' not in test_data.columns or 'category' not in test_data.columns:
            raise ValueError("The test dataset must contain 'text' and 'category' columns.")

        samples = test_data.head(3)
        
        # Extract categories from the dataset
        if "categories" in global_state.datasets:
            categories = global_state.datasets["categories"].iloc[:, 0].tolist()
        else:
            st.error("Categories not found in the dataset.")
            st.stop()

        for _, row in samples.iterrows():
            text = row["text"]
            
            if method_option == "Zero-Shot":
                if model_option == "GPT-3.5 Turbo":
                    classification = classify_with_gpt_3_5_turbo_zero_shot(text, categories)
                elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                    classification = classify_with_gpt_3_5_turbo_fine_zero_shot(text, categories)
                elif model_option == "GPT-4o":
                    classification = classify_with_gpt_4o_zero_shot(text, categories)
            elif method_option == "Few-Shot":
                if model_option == "GPT-3.5 Turbo":
                    classification = classify_with_gpt_3_5_turbo_few_shot(text, categories)
                elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                    classification = classify_with_gpt_3_5_turbo_fine_few_shot(text, categories)
                elif model_option == "GPT-4o":
                    classification = classify_with_gpt_4o_few_shot(text, categories)
            
            write_results_to_csv(model_option, method_option, text, classification)
    else:
        raise ValueError("Test dataset not loaded in global state.")
