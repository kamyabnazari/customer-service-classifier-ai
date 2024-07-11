import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score
)
from services.data_service import list_result_files
import os
import numpy as np

def evaluate_classification_results(csv_file_path):
    df = pd.read_csv(csv_file_path)
    y_true = df['true_category']
    y_pred = df['category']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Calculate specificity
    with np.errstate(divide='ignore', invalid='ignore'):
        tn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
        fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
        specificity = np.divide(tn, tn + fp)
        specificity[np.isnan(specificity)] = 0

    # Calculate False Positive Rate (FPR)
    fpr = np.divide(fp, fp + tn)
    fpr[np.isnan(fpr)] = 0

    # Calculate Geometric Mean (G-Mean)
    g_mean = np.sqrt(recall * specificity.mean())

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'specificity': specificity.mean(),
        'kappa': kappa,
        'fpr': fpr.mean(),
        'g_mean': g_mean
    }
    return metrics

def evaluate_all_results(results_dir):
    result_files = list_result_files(results_dir)
    evaluations = {}

    for file_path in result_files:
        metrics = evaluate_classification_results(file_path)
        file_name = os.path.basename(file_path)
        evaluations[file_name] = metrics

    return evaluations
