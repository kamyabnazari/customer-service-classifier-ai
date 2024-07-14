import os
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

    # Calculate token totals
    total_prompt_tokens = df['prompt_tokens'].sum()
    total_completion_tokens = df['completion_tokens'].sum()
    total_tokens = total_prompt_tokens + total_completion_tokens

    # Calculate the frequency of each predicted category
    category_usage = df['category'].value_counts()

    # Identify categories in predictions not present in true categories
    unexpected_categories = category_usage.loc[~category_usage.index.isin(y_true.unique())]

    metrics = compute_metrics(df, y_true, y_pred, labels)
    metrics.update({
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_tokens': total_tokens,
        'unexpected_categories': unexpected_categories
    })
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

    missing_categories = {}
    for label, details in class_report.items():
        if isinstance(details, dict) and details.get('support') == 0:
            missing_categories[label] = details

    return {
        'y_true': y_true, 'y_pred': y_pred, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'confusion_matrix': conf_matrix, 'classification_report': class_report,
        'specificity': np.nanmean(specificity), 'kappa': kappa, 'fpr': np.nanmean(fpr), 'g_mean': g_mean,
        'labels': labels, 'texts': texts, 'missing_categories': missing_categories
    }

def calculate_specificity_fpr(conf_matrix):
    tn = np.sum(conf_matrix) - np.sum(conf_matrix, axis=0) - np.sum(conf_matrix, axis=1) + np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)
    tn_safe = np.where((tn + fp) == 0, 1, tn)
    fp_safe = np.where((tn + fp) == 0, 1, fp)
    specificity = np.divide(tn_safe, tn_safe + fp_safe)
    fpr = np.divide(fp_safe, tn_safe + fp_safe)
    return specificity, fpr

def generate_plot(fig, filename, original_filename):
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

def escape_latex_special_chars(text):
    """Escape special LaTeX characters in strings."""
    # Add other special characters as needed
    char_map = {
        '_': r'\_',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '&': r'\&',
        '{': r'\{',
        '}': r'\}',
    }
    for char, escaped_char in char_map.items():
        text = text.replace(char, escaped_char)
    return text

def generate_table(df, filename, original_filename):
    """Convert DataFrame to LaTeX and create a download link with a unique file name."""
    tex_directory = "./customer_service_classifier_ai_data/evaluation/tables"
    
    if not os.path.exists(tex_directory):
        os.makedirs(tex_directory)
    
    # Clean and prepare the file name
    base_name = os.path.splitext(original_filename)[0]
    clean_name = base_name.replace("classification_results_", "").replace("_", "-").replace(".", "-")
    clean_name = '-'.join(filter(None, clean_name.split('-')))
    new_filename = f"table-{filename.replace('.tex', '').replace('_', '-')}-{clean_name}.tex"
    path = os.path.join(tex_directory, new_filename)
    
    # Format the caption text: replace underscores, capitalize first letters
    formatted_caption = " ".join([word.capitalize() for word in filename.replace('.tex', '').replace('_', ' ').split()])
    clean_caption = " ".join([word.capitalize() for word in clean_name.split('-')])
    
    # Determine the correct column format based on the number of DataFrame columns
    column_format = " ".join(["X"] * len(df.columns))

    # Escape special LaTeX characters in DataFrame entries
    df = df.map(lambda x: escape_latex_special_chars(x) if isinstance(x, str) else x)

    # Format DataFrame for LaTeX
    df = df.map(lambda x: f"\\num{{{x}}}" if isinstance(x, (int, float)) else x)

    # Generate LaTeX table without custom headers
    latex_table = df.to_latex(
        index=False,
        escape=False,
        sparsify=True,
        multirow=True,
        multicolumn=True,
        bold_rows=True,
        longtable=False,
        column_format=column_format
       )

    # Extract headers from DataFrame and format them
    headers = " & ".join([f"\\textbf{{{col}}}" for col in df.columns])
    latex_table = latex_table.split('\n')
    start = latex_table.index("\\toprule") + 1
    end = latex_table.index("\\midrule", start)
    latex_table = latex_table[:start] + [f"{headers} \\\\"] + latex_table[end:]
    latex_table = "\n".join(latex_table)
    latex_table = latex_table.replace("\\begin{tabular}", "\\begin{tabularx}{\\textwidth}")
    latex_table = latex_table.replace("\\end{tabular}", "\\end{tabularx}")
    
    # Wrap the table with the table environment
    latex_str = f"""
    \\begin{{table}}[!ht]
        \\centering
        {latex_table}
        \\caption{{{clean_caption} {formatted_caption}}}
        \\label{{tab:{clean_name}-{filename.replace('.tex', '').replace('_', '-')}}}
    \\end{{table}}
    """

    # Save the LaTeX string to file
    with open(path, 'w') as file:
        file.write(latex_str)

def plot_confusion_matrix(conf_matrix, labels, original_filename, show=True):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Grenzen für die Segmente definieren
    boundaries = [np.min(conf_matrix)] + list(np.linspace(np.min(conf_matrix), np.max(conf_matrix), num=4)) + [np.max(conf_matrix) + 1]
    cmap = plt.get_cmap('viridis', len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # pcolormesh für vektorbasierte, nicht gerasterte Ausgabe verwenden
    cax = ax.pcolormesh(conf_matrix, cmap=cmap, norm=norm, edgecolors='k', linewidth=0.01)
    cbar = plt.colorbar(cax, spacing='proportional', ticks=boundaries)
    cbar.set_label('Häufigkeit der Vorhersagen', rotation=270, labelpad=20, fontsize=18)

    # Ticks und Labels für Klarheit setzen
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title('Konfusionsmatrix', fontsize=16)
    plt.xlabel('Vorhergesagt (Vorhergesagte Kategorie)', fontsize=14)
    plt.ylabel('Wahr (Wahre Kategorie)', fontsize=14)
    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        return generate_plot(fig, 'confusion_matrix.pgf', original_filename)

metrics_translations = {
    'precision': 'Präzision',
    'recall': 'Erinnerungswert',
    'f1_score': 'F1-Wert'
}

def plot_classification_report(class_report, original_filename, show=True):
    report_df = pd.DataFrame(class_report).transpose()
    report_df.drop(['support'], axis=1, inplace=True)
    
    # Translate columns for the plot using the dictionary
    report_df.columns = [metrics_translations.get(col, col) for col in report_df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    report_df.plot(kind='bar', ax=ax)
    plt.title('Klassifikationsbericht', fontsize=16)
    plt.xlabel('Absichten', fontsize=10)
    plt.ylabel('Werte', fontsize=14)
    ax.set_xticklabels([])
    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        return generate_plot(fig, 'classification_report.pgf', original_filename)

def plot_class_distribution(y_true, y_pred, original_filename, show=True):
    actual_counts = pd.Series(y_true).value_counts()
    predicted_counts = pd.Series(y_pred).value_counts()
    df = pd.DataFrame({'Tatsächlich': actual_counts, 'Vorhergesagt': predicted_counts}).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind='bar', ax=ax)
    plt.title('Klassenverteilung - Tatsächlich vs. Vorhergesagt', fontsize=16)
    plt.xlabel('Absichten', fontsize=10)
    plt.ylabel('Häufigkeit', fontsize=14)
    ax.set_xticklabels([])
    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        return generate_plot(fig, 'class_distribution.pgf', original_filename)

def plot_text_length_analysis(texts, y_true, y_pred, original_filename, show=True):
    text_lengths = [len(text.split()) for text in texts]
    is_correct = np.array(y_true) == np.array(y_pred)
    df = pd.DataFrame({'Textlänge': text_lengths, 'Korrekt': is_correct})
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['blue', 'green']
    df.boxplot(column='Textlänge', by='Korrekt', ax=ax, boxprops=dict(color=colors[0]), whiskerprops=dict(color=colors[1]))
    plt.title('Textlänge vs. Klassifikationsgenauigkeit', fontsize=16)
    plt.xlabel('Ist Klassifikation korrekt?', fontsize=14)
    plt.ylabel('Textlänge', fontsize=14)
    plt.xticks([1, 2], ['Falsch', 'Richtig'], fontsize=10)
    plt.suptitle('')
    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        return generate_plot(fig, 'text_length_analysis.pgf', original_filename)

def evaluate_all_results(results_dir):
    result_files = os.listdir(results_dir)
    evaluations = {}
    token_summaries = []
    for file_path in result_files:
        full_path = os.path.join(results_dir, file_path)
        metrics = evaluate_classification_results(full_path)
        evaluations[file_path] = metrics

        # Collect token data for summary
        token_data = {
            'Model': custom_label(file_path),
            'Total Prompt Tokens': metrics['total_prompt_tokens'],
            'Total Completion Tokens': metrics['total_completion_tokens'],
            'Total Tokens': metrics['total_tokens']
        }
        token_summaries.append(token_data)
    
    return evaluations, pd.DataFrame(token_summaries).set_index('Model')

def evaluate_single_result(file_path):
    if os.path.exists(file_path):
        metrics = evaluate_classification_results(file_path)
        file_name = os.path.basename(file_path)
        return {file_name: metrics}
    else:
        raise FileNotFoundError("The specified file does not exist.")

def custom_label(file_name):
    if "few_shot" in file_name:
        return "GPT 3.5 Turbo - Few Shot" if "fine_few_shot" not in file_name else "GPT 3.5 Turbo Fine Tuned - Few Shot"
    elif "zero_shot" in file_name:
        return "GPT 3.5 Turbo - Zero Shot" if "fine_zero_shot" not in file_name else "GPT 3.5 Turbo Fine Tuned - Zero Shot"
    elif "fine_no_prompting" in file_name:
        return "GPT 3.5 Turbo Fine Tuned - Kein Prompting"
    return "Unknown Configuration"

translations = {
    'Total Prompt Tokens': 'Anzahl Prompt-Tokens',
    'Total Completion Tokens': 'Anzahl Completion-Tokens',
    'Total Tokens': 'Gesamtanzahl Tokens',
    'Accuracy': 'Genauigkeit',
    'Precision': 'Präzision',
    'Recall': 'Erinnerungswert',
    'F1 Score': 'F1-Wert',
    'Kappa': 'Kappa',
    'Specificity': 'Spezifität',
    'FPR': 'FPR',
    'G-Mean': 'G-Mittelwert'
}

def plot_token_comparisons(token_df, original_filename, show=True):
    fig, ax = plt.subplots(figsize=(15, 10))
    # Translate column names for plotting
    translated_columns = [translations[col] for col in ['Total Prompt Tokens', 'Total Completion Tokens', 'Total Tokens']]
    token_df.columns = translated_columns
    token_df[translated_columns].plot(kind='bar', ax=ax)
    plt.title('Vergleich der Token-Anzahlen über verschiedene Ergebnisse', fontsize=16)
    plt.xlabel('Konfiguration', fontsize=14)
    plt.ylabel('Anzahl der Tokens', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Token-Typen')
    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        generate_plot(fig, 'token_comparisons.pgf', original_filename)

def plot_performance_metrics(evaluations, original_filename, show=True):
    metrics_data = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Kappa': [],
        'Specificity': [],
        'FPR': [],
        'G-Mean': []
    }
    
    labels = [custom_label(file_name) for file_name in evaluations.keys()]
    for metrics in evaluations.values():
        for key in metrics_data:
            formatted_key = key.lower().replace(' ', '_').replace('-', '_')
            metrics_data[key].append(metrics[formatted_key])

    fig, ax = plt.subplots(figsize=(15, 10))
    x = np.arange(len(labels))
    width = 0.1
    n = len(metrics_data)

    for i, (metric, values) in enumerate(metrics_data.items()):
        ax.bar(x - (n/2 - i) * width, values, width, label=metric)

    ax.set_ylabel('Wertungen')
    ax.set_title('Vergleichende Leistungsmetriken')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(title='Metriken')
    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        # Convert the dictionary to DataFrame for output
        metrics_df = pd.DataFrame(metrics_data, index=labels)
        generate_plot(fig, 'performance_metrics.pgf', original_filename)
        return metrics_df