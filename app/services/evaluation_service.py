import os
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

def evaluate_classification_results(csv_file_path):
    df = pd.read_csv(csv_file_path)
    y_true = df['true_category']
    y_pred = df['category']

    labels = np.unique(np.concatenate((y_true, y_pred)))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    specificity, fpr = calculate_specificity_fpr(conf_matrix)
    g_mean = np.sqrt(recall * np.nanmean(specificity))

    metrics = {
        'y_true': y_true,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'specificity': np.nanmean(specificity),
        'kappa': kappa,
        'fpr': np.nanmean(fpr),
        'g_mean': g_mean,
        'labels': labels
    }
    return metrics

def calculate_specificity_fpr(conf_matrix):
    tn = np.sum(conf_matrix) - np.sum(conf_matrix, axis=0) - np.sum(conf_matrix, axis=1) + np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)
    tn_safe = np.where((tn + fp) == 0, 1, tn)
    fp_safe = np.where((tn + fp) == 0, 1, fp)
    specificity = np.divide(tn_safe, tn_safe + fp_safe)
    fpr = np.divide(fp_safe, tn_safe + fp_safe)
    return specificity, fpr

def plot_confusion_matrix(conf_matrix, labels):
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap='Blues')
    plt.colorbar(cax)

    # Add annotations to each cell
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > conf_matrix.max() / 2 else 'black')

    # Fix tick labels issue by setting ticks first
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

def plot_classification_report(class_report):
    report_df = pd.DataFrame(class_report).transpose()
    report_df.drop(['support'], axis=1, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    report_df.plot(kind='bar', ax=ax)
    plt.title('Classification Report')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    st.pyplot(fig)  # Show the plot in Streamlit

def plot_class_distribution(y_true, y_pred):
    actual_counts = pd.Series(y_true).value_counts()
    predicted_counts = pd.Series(y_pred).value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    df = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts}).fillna(0)
    df.plot(kind='bar', ax=ax)
    plt.title('Class Distribution - Actual vs. Predicted')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    st.pyplot(fig)  # Show the plot in Streamlit

def evaluate_all_results(results_dir):
    result_files = os.listdir(results_dir)
    evaluations = {}
    for file_path in result_files:
        metrics = evaluate_classification_results(os.path.join(results_dir, file_path))
        file_name = os.path.basename(file_path)
        evaluations[file_name] = metrics
    return evaluations
