# Import necessary packages
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from services.evaluation_service import (evaluate_all_results,
                                         plot_confusion_matrix,
                                         plot_classification_report,
                                         plot_class_distribution,
                                         plot_text_length_analysis
                                         )
from services.data_service import generate_download_link

# Dictionary for metric explanations
metric_explanations = {
    'Accuracy': 'The ratio of correctly predicted observations to the total observations.',
    'Precision': 'The ratio of correctly predicted positive observations to the total predicted positives.',
    'Recall': 'The ratio of correctly predicted positive observations to all actual positives.',
    'F1 Score': 'The weighted average of Precision and Recall.',
    'Specificity': 'The proportion of actual negatives that are correctly identified.',
    'Kappa': 'Measures inter-rater reliability for categorical items corrected for chance.',
    'False Positive Rate': 'The proportion of actual negatives that are incorrectly classified as positives.',
    'G-Mean': 'The geometric mean of sensitivity and specificity, particularly useful for imbalanced datasets.'
}

st.title("Evaluation")

# Dropdown to select the directory
directories = ['./customer_service_classifier_ai_data/results/automated', './customer_service_classifier_ai_data/results/manual',]
results_dir = st.selectbox('Select the directory path for evaluation results:', directories)

if st.button('Evaluate Results'):
    evaluations = evaluate_all_results(results_dir)
    if evaluations:
        st.header("Evaluation Metrics for Each Result File")
        for file_name, metrics in evaluations.items():
            st.write(f"Results for: {file_name}")
            
            # Summary Metrics Table
            summary_metrics = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'Kappa', 'False Positive Rate', 'G-Mean'],
                'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['specificity'], metrics['kappa'], metrics['fpr'], metrics['g_mean']],
                'Explanation': [metric_explanations[m] for m in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'Kappa', 'False Positive Rate', 'G-Mean']]
            })
            st.dataframe(summary_metrics, use_container_width=True)
            
            # Detailed Classification Report
            st.write("**Detailed Classification Report:**")
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
            download_link = generate_download_link(report_df, f"classification_report_{file_name}")
            st.markdown(download_link, unsafe_allow_html=True)

            # Confusion Matrix Visualization
            st.write("**Confusion Matrix:**")
            plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'])

            # Additional Visualizations as discussed
            plot_classification_report(metrics['classification_report'])
            plot_class_distribution(metrics['y_true'], metrics['y_pred'])
            plot_text_length_analysis(metrics['texts'], metrics['y_true'], metrics['y_pred'])

            st.divider()
    else:
        st.error("No evaluation data found. Please check the directory path and ensure it contains result files.")
