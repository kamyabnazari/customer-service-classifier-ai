import base64
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from services.evaluation_service import evaluate_all_results
import io

st.title("Evaluation")

results_dir = './customer_service_classifier_ai_data/results/automated'
evaluations = evaluate_all_results(results_dir)

st.header("Evaluation Metrics for Each Result File")

def generate_download_link(data, file_name):
    buffer = io.StringIO()
    data.to_csv(buffer, index=True)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download CSV</a>'
    return href

for file_name, metrics in evaluations.items():
    st.subheader(f"Results for {file_name}")

    summary_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'Kappa', 'False Positive Rate', 'G-Mean'],
        'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['specificity'], metrics['kappa'], metrics['fpr'], metrics['g_mean']]
    })

    st.write("**Summary Metrics:**")
    st.table(summary_metrics)

    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(metrics['confusion_matrix'], cmap='Blues')
    plt.colorbar(cax)
    ax.set_xticks(range(len(metrics['labels'])))
    ax.set_yticks(range(len(metrics['labels'])))
    ax.set_xticklabels(metrics['labels'], rotation=45, ha='right')
    ax.set_yticklabels(metrics['labels'])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    for (i, j), val in np.ndenumerate(metrics['confusion_matrix']):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > metrics['confusion_matrix'].max()/2 else 'black')

    st.pyplot(fig)

    st.write("**Detailed Classification Report:**")
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df)

    download_link = generate_download_link(report_df, f"classification_report_{file_name}")
    st.markdown(download_link, unsafe_allow_html=True)

    st.divider()
