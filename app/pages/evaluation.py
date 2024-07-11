# Import necessary packages
import streamlit as st
import pandas as pd
from services.evaluation_service import (evaluate_all_results, get_image_download_link, get_table_download_link,
                                         show_confusion_matrix, save_confusion_matrix,
                                         show_classification_report, save_classification_report,
                                         show_class_distribution, save_class_distribution,
                                         show_text_length_analysis, save_text_length_analysis)
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
            st.markdown(get_table_download_link(report_df, "classification_report.tex"), unsafe_allow_html=True)

            # Confusion Matrix Visualization
            st.write("**Confusion Matrix:**")
            show_confusion_matrix(metrics['confusion_matrix'], metrics['labels'])
            conf_matrix_path = save_confusion_matrix(metrics['confusion_matrix'], metrics['labels'])
            st.markdown(get_image_download_link(conf_matrix_path, "Download confusion matrix"), unsafe_allow_html=True)
            
            # Additional Visualizations as discussed
            st.write("**Classification Report:**")
            show_classification_report(metrics['classification_report'])
            class_report_path = save_classification_report(metrics['classification_report'])
            st.markdown(get_image_download_link(class_report_path, "Download classification report"), unsafe_allow_html=True)

            st.write("**Class Distribution:**")
            show_class_distribution(metrics['y_true'], metrics['y_pred'])
            class_distribution_path = save_class_distribution(metrics['y_true'], metrics['y_pred'])
            st.markdown(get_image_download_link(class_distribution_path, "Download class distribution"), unsafe_allow_html=True)

            st.write("**Text Length Analysis:**")
            show_text_length_analysis(metrics['texts'], metrics['y_true'], metrics['y_pred'])
            text_length_path = save_text_length_analysis(metrics['texts'], metrics['y_true'], metrics['y_pred'])
            st.markdown(get_image_download_link(text_length_path, "Download text length analysis"), unsafe_allow_html=True)

            st.divider()
    else:
        st.error("No evaluation data found. Please check the directory path and ensure it contains result files.")
