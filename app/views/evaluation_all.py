import streamlit as st
from services.evaluation_service import evaluate_all_results, plot_performance_metrics, plot_token_comparisons

st.title("Evaluation All")

# User selects a directory of result files
results_dir = st.selectbox('Select the directory path for evaluation results:', ['./customer_service_classifier_ai_data/results/automated'])

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("views/home.py")
else:
    if st.button('Compare Results', use_container_width=True):
        evaluations, token_df = evaluate_all_results(results_dir)
        
        st.write("**Token Comparison Across Results:**")
        plot_token_comparisons(token_df, 'token_comparisons.pgf', show=True)
        plot_token_comparisons(token_df, 'token_comparisons.pgf', show=False)

        # Optionally, display the raw data table as well
        st.write("**Raw Token Data:**")
        st.table(token_df)

        st.write("**Performance Metrics:**")
        plot_performance_metrics(evaluations, 'performance_metrics.pgf', show=True)
        plot_performance_metrics(evaluations, 'performance_metrics.pgf', show=False)