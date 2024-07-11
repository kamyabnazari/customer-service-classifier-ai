import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score
)
import os
import numpy as np

# Evaluate classification results from a CSV file
def evaluate_classification_results(csv_file_path):
    df = pd.read_csv(csv_file_path)
    y_true = df['true_category']
    y_pred = df['category']

    # Calculate standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Calculate additional metrics such as specificity and FPR
    tn, fp = calculate_specificity_fpr(conf_matrix)
    specificity = np.mean(tn / (tn + fp))
    fpr = np.mean(fp / (fp + tn))
    g_mean = np.sqrt(recall * specificity)  # Adjusted to correctly calculate G-Mean

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'specificity': specificity,
        'kappa': kappa,
        'fpr': fpr,
        'g_mean': g_mean,
        'labels': np.unique(y_true)
    }
    return metrics

def calculate_specificity_fpr(conf_matrix):
    # Calculates True Negatives and False Positives for Specificity and FPR calculations
    tn = conf_matrix.sum() - (conf_matrix.sum(axis=0) + conf_matrix.sum(axis=1) - np.diag(conf_matrix))
    fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    return tn, fp

def evaluate_all_results(results_dir):
    result_files = os.listdir(results_dir)  # Simplified file listing
    evaluations = {}
    for file_path in result_files:
        metrics = evaluate_classification_results(os.path.join(results_dir, file_path))
        file_name = os.path.basename(file_path)
        evaluations[file_name] = metrics
    return evaluations
