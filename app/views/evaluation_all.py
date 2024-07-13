import streamlit as st
from services.evaluation_service import evaluate_all_results, plot_performance_metrics, plot_token_comparisons
st.title("Comparative Evaluation Across Multiple Results")

# User selects a directory of result files
results_dir = st.selectbox('Select the directory path for evaluation results:', ['./customer_service_classifier_ai_data/results/automated'])

if st.button('Compare Results'):
    evaluations, token_df = evaluate_all_results(results_dir)
    if not evaluations:
        st.error("No evaluation data found in the selected directory.")
    else:
        st.write("### Token Comparison Across Results")
        plot_token_comparisons(token_df, 'token_comparisons.pgf', show=True)
        plot_token_comparisons(token_df, 'token_comparisons.pgf', show=False)

        # Optionally, display the raw data table as well
        st.write("### Raw Token Data")
        st.table(token_df)
    
    plot_performance_metrics(evaluations, 'performance_metrics.pgf', show=True)
    plot_performance_metrics(evaluations, 'performance_metrics.pgf', show=False)