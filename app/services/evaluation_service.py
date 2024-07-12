import os
import base64
import matplotlib
from matplotlib.colors import BoundaryNorm
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

# Configure Matplotlib to use LaTeX for rendering
matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def evaluate_classification_results(csv_file_path):
    df = pd.read_csv(csv_file_path)
    y_true = df['true_category']
    y_pred = df['category']
    labels = np.unique(np.concatenate((y_true, y_pred)))
    metrics = compute_metrics(df, y_true, y_pred, labels)
    return metrics

def compute_metrics(df, y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    specificity, fpr = calculate_specificity_fpr(conf_matrix)
    g_mean = np.sqrt(recall * np.nanmean(specificity))
    texts = df['request'].tolist()
    
    return {'y_true': y_true, 'y_pred': y_pred, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1, 'confusion_matrix': conf_matrix, 'classification_report': class_report,
            'specificity': np.nanmean(specificity), 'kappa': kappa, 'fpr': np.nanmean(fpr), 'g_mean': g_mean,
            'labels': labels, 'texts': texts}

def calculate_specificity_fpr(conf_matrix):
    tn = np.sum(conf_matrix) - np.sum(conf_matrix, axis=0) - np.sum(conf_matrix, axis=1) + np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)
    tn_safe = np.where((tn + fp) == 0, 1, tn)
    fp_safe = np.where((tn + fp) == 0, 1, fp)
    specificity = np.divide(tn_safe, tn_safe + fp_safe)
    fpr = np.divide(fp_safe, tn_safe + fp_safe)
    return specificity, fpr

def get_plot_download_link(fig, filename, original_filename, caption="Download"):
    """Save the plot as a PGF file and create a download link."""
    pgf_directory = "./customer_service_classifier_ai_data/evaluation/pgf"
    tex_directory = "./customer_service_classifier_ai_data/evaluation/figures"
    
    if not os.path.exists(pgf_directory):
        os.makedirs(pgf_directory)
    if not os.path.exists(tex_directory):
        os.makedirs(tex_directory)
    
    # Clean and prepare the file name
    base_name = os.path.splitext(original_filename)[0]
    clean_name = base_name.replace("classification_results_", "").replace("_", "-").replace(".", "-")
    # Adjust the order of clean name and filename
    pgf_filename = f"pgf-{clean_name}-{filename.replace('.pgf', '').replace('_', '-')}.pgf"
    figure_filename = f"figure-{clean_name}-{filename.replace('.pgf', '').replace('_', '-')}.tex"
    pgf_path = os.path.join(pgf_directory, pgf_filename)
    tex_path = os.path.join(tex_directory, figure_filename)

    # Save the figure to the PGF path
    fig.savefig(pgf_path, format='pgf')
    plt.close(fig)

    # Format the caption text: replace underscores, capitalize first letters, and remove extensions
    formatted_caption = " ".join([word.capitalize() for word in filename.replace('.pgf', '').replace('_', ' ').split()])
    clean_caption = " ".join([word.capitalize() for word in clean_name.split('-')])

    # Create the LaTeX file at the TeX path
    with open(tex_path, "w") as tex_file:
        tex_file.write(
            "\\begin{figure}[ht]\n"
            "    \\centering\n"
            "    \\resizebox{1.0\\textwidth}{!}{\n"
            f"        \\input{{resources/pgf/{pgf_filename}}}\n"
            "    }\n"
            f"    \\caption{{{clean_caption} {formatted_caption}}}\n"
            f"    \\label{{fig:{clean_name}-{filename.replace('.pgf', '').replace('_', '-')}}}\n"
            "\\end{figure}\n"
        )

    # Create the download link for the PGF file
    with open(pgf_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pgf_filename}">{caption}</a>'
    return href

def get_table_download_link(df, filename, original_filename, caption="Download LaTeX table"):
    """Convert DataFrame to LaTeX and create a download link with a unique file name."""
    tex_directory = "./customer_service_classifier_ai_data/evaluation/tables"
    
    if not os.path.exists(tex_directory):
        os.makedirs(tex_directory)
    
    # Clean and prepare the file name
    base_name = os.path.splitext(original_filename)[0]
    clean_name = base_name.replace("classification_results_", "").replace("_", "-").replace(".", "-")
    clean_name = '-'.join(filter(None, clean_name.split('-')))
    new_filename = f"table-{clean_name}-{filename.replace('.tex', '').replace('_', '-')}.tex"
    path = os.path.join(tex_directory, new_filename)
    
    # Generate LaTeX table without custom headers
    latex_table = df.to_latex(index=False, column_format="X l", 
                              bold_rows=False, escape=False, longtable=False)

    # Extract headers from DataFrame and format them
    headers = " & ".join([f"\\textbf{{{col}}}" for col in df.columns])
    header_latex = f"\\toprule\n{headers} \\\\\n\\midrule"

    # Replace the default headers with custom formatted headers
    latex_table = latex_table.split('\n')
    # Remove the existing header line, which is typically between the first \toprule and the first \midrule
    start = latex_table.index("\\toprule") + 1
    end = latex_table.index("\\midrule", start)
    latex_table = latex_table[:start] + [f"{headers} \\\\"] + latex_table[end:]

    # Rejoin the table and wrap with tabularx environment
    latex_table = "\n".join(latex_table)
    latex_table = latex_table.replace("\\begin{tabular}{Xl}", "\\begin{tabularx}{\\textwidth}{X l}")
    latex_table = latex_table.replace("\\end{tabular}", "\\end{tabularx}")
    
    # Wrap the table with the table environment
    latex_str = f"""
                \\begin{{table}}[!ht]
                    \\centering
                    {latex_table}
                    \\caption{{{caption}}}
                    \\label{{tab:{clean_name}}}
                \\end{{table}}
                """
    
    # Save the LaTeX string to file
    with open(path, 'w') as file:
        file.write(latex_str)
    
    # Create the download link
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    href = f'<a href="data:file/tex;base64,{b64}" download="{new_filename}">Download LaTeX table</a>'
    
    return href

def plot_confusion_matrix(conf_matrix, labels, original_filename, show=True):
    fig, ax = plt.subplots()
    # Setup the plot as before
    boundaries = [np.min(conf_matrix) - 1] + list(np.linspace(np.min(conf_matrix), np.max(conf_matrix), num=4)) + [np.max(conf_matrix) + 1]
    cmap = plt.get_cmap('Blues', len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    cax = ax.pcolormesh(conf_matrix, cmap=cmap, norm=norm, edgecolors='k', linewidth=2)
    plt.colorbar(cax, spacing='proportional')
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j + 0.5, i + 0.5, f'{val}', ha='center', va='center', color='white' if val > conf_matrix.max() / 2 else 'black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if show:
        st.pyplot(fig)
    else:
        return get_plot_download_link(fig, 'confusion_matrix.pgf', original_filename)

def plot_classification_report(class_report, original_filename, show=True):
    report_df = pd.DataFrame(class_report).transpose()
    report_df.drop(['support'], axis=1, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    report_df.plot(kind='bar', ax=ax)
    plt.title('Classification Report')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    if show:
        st.pyplot(fig)
    else:
        return get_plot_download_link(fig, 'classification_report.pgf', original_filename)

def plot_class_distribution(y_true, y_pred, original_filename, show=True):
    actual_counts = pd.Series(y_true).value_counts()
    predicted_counts = pd.Series(y_pred).value_counts()
    df = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts}).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind='bar', ax=ax)
    plt.title('Class Distribution - Actual vs. Predicted')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    if show:
        st.pyplot(fig)
    else:
        return get_plot_download_link(fig, 'class_distribution.pgf', original_filename)

def plot_text_length_analysis(texts, y_true, y_pred, original_filename, show=True):
    text_lengths = [len(text.split()) for text in texts]
    is_correct = np.array(y_true) == np.array(y_pred)
    df = pd.DataFrame({'Text Length': text_lengths, 'Correct': is_correct})
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Text Length', by='Correct', ax=ax)
    plt.title('Text Length vs. Classification Correctness')
    plt.xlabel('Is Classification Correct?')
    plt.ylabel('Text Length')
    plt.xticks([1, 2], ['Incorrect', 'Correct'])
    if show:
        st.pyplot(fig)
    else:
        return get_plot_download_link(fig, 'text_length_analysis.pgf', original_filename)

def evaluate_all_results(results_dir):
    result_files = os.listdir(results_dir)
    evaluations = {}
    for file_path in result_files:
        full_path = os.path.join(results_dir, file_path)
        metrics = evaluate_classification_results(full_path)
        file_name = os.path.basename(file_path)
        evaluations[file_name] = metrics
    return evaluations

def evaluate_single_result(file_path):
    if os.path.exists(file_path):
        metrics = evaluate_classification_results(file_path)
        file_name = os.path.basename(file_path)
        return {file_name: metrics}
    else:
        raise FileNotFoundError("The specified file does not exist.")