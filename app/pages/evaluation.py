import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

    st.write("**Summary Metrics:**")
    summary_metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    }
    summary_df = pd.DataFrame(summary_metrics)
    st.table(summary_df.style.format({"Value": "{:.2f}"}))

    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(metrics['confusion_matrix'], cmap='Blues')
    fig.colorbar(cax)

    # Annotate the confusion matrix
    for (i, j), val in np.ndenumerate(metrics['confusion_matrix']):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > metrics['confusion_matrix'].max()/2 else 'black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    labels = list(metrics['classification_report'].keys())[:-3]  # Remove 'accuracy', 'macro avg', and 'weighted avg'
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    
    st.pyplot(fig)

    st.write("**Detailed Classification Report:**")
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df)

    download_link = generate_download_link(report_df, f"classification_report_{file_name}")
    st.markdown(download_link, unsafe_allow_html=True)

    st.divider()
