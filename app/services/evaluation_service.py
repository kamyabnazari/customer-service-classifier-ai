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

def get_plot_path(filename):
    """Generates a file path for saving plots."""
    directory = "./customer_service_classifier_ai_data/temp_plots"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.join(directory, filename)

def get_image_download_link(img_path, caption="Download"):
    """Generate a link to download a file."""
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(img_path)}">{caption}</a>'
    return href

def get_table_download_link(df, filename="data_table.tex", caption="Download LaTeX table"):
    """Convert DataFrame to LaTeX and create a download link."""
    latex_str = df.to_latex(index=False)
    b64 = base64.b64encode(latex_str.encode()).decode("utf-8")
    href = f'<a href="data:file/tex;base64,{b64}" download="{filename}">{caption}</a>'
    return href

def plot_confusion_matrix(conf_matrix, labels, show=True):
    path = get_plot_path('confusion_matrix.pgf')
    fig, ax = plt.subplots()
    
    # Define the boundaries for your discrete colormap
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
        plt.savefig(path, format='pgf')
        plt.close(fig)
        return path
    plt.close(fig)

def plot_classification_report(class_report, show=True):
    path = get_plot_path('classification_report.pgf')
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
        plt.savefig(path, format='pgf')
        plt.close(fig)
        return path
    plt.close(fig)

def plot_class_distribution(y_true, y_pred, show=True):
    path = get_plot_path('class_distribution.pgf')
    actual_counts = pd.Series(y_true).value_counts()
    predicted_counts = pd.Series(y_pred).value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    df = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts}).fillna(0)
    df.plot(kind='bar', ax=ax)
    plt.title('Class Distribution - Actual vs. Predicted')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    if show:
        st.pyplot(fig)
    else:
        plt.savefig(path, format='pgf')
        plt.close(fig)
        return path
    plt.close(fig)

def plot_text_length_analysis(texts, y_true, y_pred, show=True):
    path = get_plot_path('text_length_analysis.pgf')
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
        plt.savefig(path, format='pgf')
        plt.close(fig)
        return path
    plt.close(fig)

def evaluate_all_results(results_dir):
    result_files = os.listdir(results_dir)
    evaluations = {}
    for file_path in result_files:
        full_path = os.path.join(results_dir, file_path)
        metrics = evaluate_classification_results(full_path)
        file_name = os.path.basename(file_path)
        evaluations[file_name] = metrics
    return evaluations
