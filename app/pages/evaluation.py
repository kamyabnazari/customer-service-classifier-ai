import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from services.evaluation_service import evaluate_all_results
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

# Predefined list of directories
directories = [
    './customer_service_classifier_ai_data/results/automated',
    './customer_service_classifier_ai_data/results/manual',
]

# Dropdown to select the directory
results_dir = st.selectbox('Select the directory path for evaluation results:', directories)
if st.button('Evaluate Results', use_container_width=True):
    evaluations = evaluate_all_results(results_dir)
    if evaluations:
        st.header("Evaluation Metrics for Each Result File")
        for file_name, metrics in evaluations.items():
            st.write(f"**Results for:** {file_name}")
            
            summary_metrics = pd.DataFrame({
                'Metric': [
                    'Accuracy',
                    'Precision',
                    'Recall',
                    'F1 Score',
                    'Specificity',
                    'Kappa',
                    'False Positive Rate',
                    'G-Mean'
                    ],
                'Value': [
                    metrics['accuracy'], 
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1_score'],
                    metrics['specificity'],
                    metrics['kappa'],
                    metrics['fpr'],
                    metrics['g_mean']
                    ],
                'Explanation': [
                    metric_explanations["Accuracy"],
                    metric_explanations["Precision"],
                    metric_explanations["Recall"],
                    metric_explanations["F1 Score"],
                    metric_explanations["Specificity"],
                    metric_explanations["Kappa"],
                    metric_explanations["False Positive Rate"],
                    metric_explanations["G-Mean"],
                ]
            })
            
            st.write("**Summary Metrics:**")
            st.dataframe(summary_metrics, use_container_width=True)
            
            st.write("**Detailed Classification Report:**")
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            download_link = generate_download_link(report_df, f"classification_report_{file_name}")
            st.markdown(download_link, unsafe_allow_html=True)

            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(metrics['confusion_matrix'], cmap='viridis')
            plt.colorbar(cax)
            ax.set_xticks(np.arange(len(metrics['labels'])))
            ax.set_yticks(np.arange(len(metrics['labels'])))
            ax.set_xticklabels(metrics['labels'], rotation=45, ha='right')
            ax.set_yticklabels(metrics['labels'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            for (i, j), val in np.ndenumerate(metrics['confusion_matrix']):
                ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > metrics['confusion_matrix'].max()/2 else 'black')
            
            st.pyplot(fig)
            
            st.divider()
    else:
        st.error("No evaluation data found. Please check the directory path and ensure it contains result files.")
