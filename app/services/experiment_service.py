import streamlit as st
from services.openai_service import (
    classify_with_gpt_3_5_turbo_zero_shot,
    classify_with_gpt_3_5_turbo_few_shot,
    classify_with_gpt_3_5_turbo_fine_zero_shot,
    classify_with_gpt_3_5_turbo_fine_few_shot,
    classify_with_gpt_3_5_turbo_fine_no_prompting
)

def classify_test_samples(global_state, model_option, method_option, temperature, classification_method):
    if "test" in global_state.datasets:
        test_data = global_state.datasets["test"]
        
        # Überprüfen, ob die erforderlichen Spalten vorhanden sind
        if 'text' not in test_data.columns or 'category' not in test_data.columns:
            raise ValueError("The test dataset must contain 'text' and 'category' columns.")

        samples = test_data # The Full Test Data
        
        # Extract categories from the dataset
        if "categories" in global_state.datasets:
            categories = global_state.datasets["categories"].iloc[:, 0].tolist()
        else:
            st.error("Categories not found in the dataset.")
            st.stop()

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for index, row in samples.iterrows():
            try:
                text = row["text"]
                true_category = row["category"]

                if method_option == "No Prompting":
                    if model_option == "GPT-3.5 Turbo Fine-Tuned":
                        classify_with_gpt_3_5_turbo_fine_no_prompting(text, temperature, classification_method, true_category)
                elif method_option == "Zero-Shot":
                    if model_option == "GPT-3.5 Turbo":
                        classify_with_gpt_3_5_turbo_zero_shot(text, categories, temperature, classification_method, true_category)
                    elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                        classify_with_gpt_3_5_turbo_fine_zero_shot(text, categories, temperature, classification_method, true_category)
                elif method_option == "Few-Shot":
                    if model_option == "GPT-3.5 Turbo":
                        classify_with_gpt_3_5_turbo_few_shot(text, categories, temperature, classification_method, true_category)
                    elif model_option == "GPT-3.5 Turbo Fine-Tuned":
                        classify_with_gpt_3_5_turbo_fine_few_shot(text, categories, temperature, classification_method, true_category)
            except Exception as e:
                st.error(f"Failed to process sample {index + 1} due to error: {e}")
                continue  # Skip the current sample or decide how you want to handle the error

            # Update progress bar and text
            progress_percentage = (index + 1) / len(samples)
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Processing sample {index + 1} of {len(samples)}")
        
        st.success("Classification completed.")
    else:
        raise ValueError("Test dataset not loaded in global state.")
