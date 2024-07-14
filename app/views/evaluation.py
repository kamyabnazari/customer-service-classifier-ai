import streamlit as st
import pandas as pd
import os
from services.evaluation_service import (
    evaluate_single_result, generate_table,
    plot_confusion_matrix, plot_classification_report,
    plot_class_distribution, plot_text_length_analysis
)

st.title("Evaluation")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("views/home.py")
else:
    # Dropdown to select the directory
    directories = [
        './customer_service_classifier_ai_data/results/automated'
    ]
    results_dir = st.selectbox('Select the directory path for evaluation results:', directories)

    # Dropdown to select a specific file within the directory
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        selected_file = st.selectbox('Select a file for evaluation:', files)
    else:
        files = []
        selected_file = None
        st.error("Selected directory is empty or does not exist.")

    if st.button('Evaluate Results', use_container_width=True) and selected_file:
        file_path = os.path.join(results_dir, selected_file)
        evaluations = evaluate_single_result(file_path)
        if evaluations:
            st.header("Evaluation Metrics for Each Result File")
            for file_name, metrics in evaluations.items():
                st.write(f"Results for: {file_name}")
                
                # Summary Metrics Table
                summary_metrics = pd.DataFrame({
                    'Metrik': ['Genauigkeit', 'Präzision', 'Erinnerungswert', 'F1-Wert', 'Spezifität', 'Kappa', 'Falsche Positive Rate', 'G-Mittelwert'],
                    'Wert': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['specificity'], metrics['kappa'], metrics['fpr'], metrics['g_mean']]
                })
                st.dataframe(summary_metrics, use_container_width=True)
                generate_table(summary_metrics, "summary_metrics_report.tex", original_filename=selected_file)

                # Detailed Classification Report
                st.write("**Detailed Classification Report:**")
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                report_df = report_df.reset_index()
                report_df.columns = ['Kategorie', 'Präzision', 'Erinnerungswert', 'F1-Wert', 'Unterstützung']
                report_df = report_df[~report_df['Kategorie'].isin(['accuracy', 'macro avg', 'weighted avg'])]
                report_df = report_df.sort_values(by='Präzision', ascending=False)
                st.dataframe(report_df, use_container_width=True)
                generate_table(report_df, "classification_report.tex", original_filename=selected_file)

                st.write("**Unexpected Category Usage Report:**")
                if 'unexpected_categories' in metrics and not metrics['unexpected_categories'].empty:
                    unexpected_categories_df = pd.DataFrame(metrics['unexpected_categories']).transpose()
                    unexpected_categories_df = unexpected_categories_df.reset_index()

                    # Transpose the DataFrame so that categories are rows
                    unexpected_categories_df = unexpected_categories_df.T
                    unexpected_categories_df.columns = ['Anzahl']
                    unexpected_categories_df = unexpected_categories_df.reset_index()
                    unexpected_categories_df.columns = ['Kategorie', 'Anzahl']

                    # Remove the first row which contains unwanted 'index' and 'count' as data
                    unexpected_categories_df = unexpected_categories_df[unexpected_categories_df['Kategorie'] != 'index']

                    st.dataframe(unexpected_categories_df, use_container_width=True)
                    generate_table(unexpected_categories_df, "unexpected_categories.tex", original_filename=file_name)

                # Confusion Matrix Visualization
                st.write("**Confusion Matrix:**")
                plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'], original_filename=file_name, show=True)
                plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'], original_filename=file_name, show=False)
                
                # Additional Visualizations as discussed
                st.write("**Classification Report:**")
                plot_classification_report(metrics['classification_report'], original_filename=file_name, show=True)
                plot_classification_report(metrics['classification_report'], original_filename=file_name, show=False)

                st.write("**Class Distribution:**")
                plot_class_distribution(metrics['y_true'], metrics['y_pred'], original_filename=file_name, show=True)
                plot_class_distribution(metrics['y_true'], metrics['y_pred'], original_filename=file_name, show=False)

                st.write("**Text Length Analysis:**")
                plot_text_length_analysis(metrics['texts'], metrics['y_true'], metrics['y_pred'], original_filename=file_name, show=True)
                plot_text_length_analysis(metrics['texts'], metrics['y_true'], metrics['y_pred'], original_filename=file_name, show=False)

                st.divider()
        else:
            st.error("No evaluation data found in the selected file.")
    else:
        st.info("Please select a file to evaluate.")
