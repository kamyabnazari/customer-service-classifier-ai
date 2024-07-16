import textwrap
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from services.evaluation_service import (
    generate_plot,
    metrics_translations,
    translations,
    custom_label
    )

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

    # Set the font size for colorbar tick labels
    cbar.ax.tick_params(labelsize=14)

    # Ticks und Labels für Klarheit setzen
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title('Konfusionsmatrix', fontsize=20, pad=20)
    plt.xlabel('Vorhergesagt (Vorhergesagte Kategorie)', fontsize=18, labelpad=20)
    plt.ylabel('Wahr (Wahre Kategorie)', fontsize=18, labelpad=20)

    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        return generate_plot(fig, 'confusion_matrix.pgf', original_filename)

def plot_classification_report(class_report, original_filename, show=True):
    report_df = pd.DataFrame(class_report).transpose()
    report_df.drop(['support'], axis=1, inplace=True)
    
    report_df.columns = [metrics_translations.get(col, col) for col in report_df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    report_df.plot(kind='bar', ax=ax)
    plt.title('Klassifikationsbericht', fontsize=20, pad=20)
    plt.xlabel('Absichten', fontsize=18, labelpad=20)
    plt.ylabel('Werte', fontsize=18, labelpad=20)
    ax.set_xticklabels([])

    # Set font size for x-axis and y-axis tick labels
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

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
    plt.title('Klassenverteilung - Tatsächlich vs. Vorhergesagt', fontsize=20, pad=20)
    plt.xlabel('Absichten', fontsize=18, labelpad=20)
    plt.ylabel('Häufigkeit', fontsize=18, labelpad=20)
    ax.set_xticklabels([])

    # Set font size for x-axis and y-axis tick labels
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

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
    plt.title('Textlänge vs. Klassifikationsgenauigkeit', fontsize=20, pad=20)
    plt.xlabel('Ist Klassifikation korrekt?', fontsize=18, labelpad=20)
    plt.ylabel('Textlänge (Wörter)', fontsize=18, labelpad=20)
    plt.xticks([1, 2], ['Falsch', 'Richtig'], fontsize=14)
    plt.suptitle('')

    # Set font size for x-axis and y-axis tick labels
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        return generate_plot(fig, 'text_length_analysis.pgf', original_filename)

def plot_token_comparisons(token_df, original_filename, show=True):
    fig, ax = plt.subplots(figsize=(15, 10))
    translated_columns = [translations[col] for col in ['Total Prompt Tokens', 'Total Completion Tokens', 'Total Tokens']]
    token_df.columns = translated_columns
    token_df[translated_columns].plot(kind='bar', ax=ax)
    plt.title('Vergleich der Token-Anzahlen über verschiedene Ergebnisse', fontsize=22, pad=20)
    plt.xlabel('Konfiguration', fontsize=20, labelpad=20)
    plt.ylabel('Anzahl der Tokens', fontsize=20, labelpad=20)
    plt.xticks(rotation=0)
    plt.legend(title='Token-Typen')

    # Set font size for x-axis and y-axis tick labels
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

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

    ax.set_ylabel('Wertungen', fontsize=20, labelpad=20)
    ax.set_title('Vergleichende Leistungsmetriken', fontsize=22, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(title='Metriken')

    # Set font size for x-axis and y-axis tick labels
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()

    if show:
        st.pyplot(fig)
    else:
        metrics_df = pd.DataFrame(metrics_data, index=labels)
        generate_plot(fig, 'performance_metrics.pgf', original_filename)
        return metrics_df

def plot_training_metrics(csv_file, original_filename=None, show=True):
    df = pd.read_csv(csv_file)

    # Plot the training and validation metrics
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_xlabel('Schritt (jede 10.)', fontsize=20, labelpad=20)
    ax.set_ylabel('Trainingsverlust', fontsize=20, labelpad=20)

    df = df.iloc[::10, :]

    ax.plot(df['step'], df['train_loss'], label='Train Loss')
    ax.grid(True)

    # Add title and labels
    plt.title('Trainings- und Validierungsmetriken über Schritte (jede 10.)', fontsize=22, pad=20)

    # Set font size for x-axis and y-axis tick labels
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    fig.tight_layout()

    # Show legend
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    if show:
        st.pyplot(fig)
    else:
        generate_plot(fig, 'training_metrics.pgf', original_filename)