import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from services.evaluation_service import evaluate_all_results

st.title("Evaluation")

results_dir = './customer_service_classifier_ai_data/results'
evaluations = evaluate_all_results(results_dir)

st.header("Evaluation Metrics for Each Result File")
for file_name, metrics in evaluations.items():
    st.subheader(f"Results for {file_name}")

    st.write(f"**Accuracy:** {metrics['accuracy']:.2f}")
    st.write(f"**Precision:** {metrics['precision']:.2f}")
    st.write(f"**Recall:** {metrics['recall']:.2f}")
    st.write(f"**F1 Score:** {metrics['f1_score']:.2f}")

    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    cax = ax.matshow(metrics['confusion_matrix'], cmap='Blues')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    st.write("**Detailed Classification Report:**")
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df)

    st.divider()
