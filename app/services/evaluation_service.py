import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from services.data_service import list_result_files
import os

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

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
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
