import streamlit as st
from services.evaluation_service import evaluate_all_results

st.title("Evaluation")

results_dir = './customer_service_classifier_ai_data/results'
evaluations = evaluate_all_results(results_dir)

st.header("Evaluation Metrics for Each Result File")
for file_name, metrics in evaluations.items():
    st.write(f"{file_name}")
    st.write(f"**Accuracy:** {metrics['accuracy']}")
    st.write(f"**Precision:** {metrics['precision']}")
    st.write(f"**Recall:** {metrics['recall']}")
    st.write(f"**F1 Score:** {metrics['f1_score']}")
    st.write("**Confusion Matrix:**")
    st.write(metrics['confusion_matrix'])
    st.divider()
