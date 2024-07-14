import streamlit as st
from services.evaluation_service import (
    evaluate_all_results,
    generate_table
    )
from services.plot_service import (
    plot_performance_metrics,
    plot_token_comparisons
    )

st.title("Evaluation All")

# Check if a dataset is loaded
if not st.session_state.get("dataset_loaded", False):
    st.warning("Please load a dataset first on the Home page.")
    if st.button("Go to Home"):
        st.switch_page("views/home.py")
else:
    # User selects a directory of result files
    results_dir = st.selectbox('Select the directory path for evaluation results:', ['./customer_service_classifier_ai_data/results/automated'])

    if st.button('Compare Results', use_container_width=True):
        evaluations, token_df = evaluate_all_results(results_dir)
        
        st.write("**Token Comparison Plot:**")
        plot_token_comparisons(token_df, 'token_comparisons.pgf', show=True)
        plot_token_comparisons(token_df, 'token_comparisons.pgf', show=False)

        st.divider()

        # Optionally, display the raw data table as well
        st.write("**Token Comparison Data:**")
        st.dataframe(token_df, use_container_width=True)
        generate_table(token_df, "token_comparison.tex", original_filename="all-evaluations")

        st.divider()

        st.write("**Performance Metrics:**")
        plot_performance_metrics(evaluations, 'performance_metrics.pgf', show=True)
        metrics_performance = plot_performance_metrics(evaluations, 'performance_metrics.pgf', show=False)

        st.divider()

        st.write("**Performance Metrics Data:**")
        metrics_performance = metrics_performance.reset_index()
        metrics_performance.columns = ['Model', 'Genauigkeit', 'Präzision', 'Erinnerungswert', 'F1-Wert', 'Kappa', 'Spezifität', 'Falsche Positive Rate', 'G-Mittelwert']
        st.dataframe(metrics_performance, use_container_width=True)
        generate_table(metrics_performance, "performance_metrics.tex", original_filename="all-evaluations")