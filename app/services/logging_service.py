import os
import csv
from datetime import datetime, timezone

# Basis-Log-Verzeichnis
base_log_dir = './customer_service_classifier_ai_data/logs'
base_results_dir = './customer_service_classifier_ai_data/results'

def get_log_file_path(model, classification_type):
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    log_dir = os.path.join(base_log_dir, f'{model}_{classification_type}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = f'classification_logs_{current_date}.txt'
    return os.path.join(log_dir, log_file_name)

def write_log_to_file(model, classification_type, system_message, input_text, classification, usage):
    log_file_path = get_log_file_path(model, classification_type)
    log_message = (
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"Model: {model}\n"
        f"Classification Type: {classification_type}\n"
        f"System Message: {system_message}\n"
        f"Input Text: {input_text}\n"
        f"Classification: {classification}\n"
        f"Usage: (Prompt Tokens: {usage.prompt_tokens}, Completion Tokens: {usage.completion_tokens}, Total Tokens: {usage.total_tokens})\n"
        "----------------------------------------\n"
    )
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_message)

def get_csv_file_path(model, classification_type):
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    results_dir = os.path.join(base_results_dir, f'{model}_{classification_type}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    csv_file_name = f'classification_results_{current_date}.csv'
    return os.path.join(results_dir, csv_file_name)

def write_results_to_csv(model, classification_type, input_text, classification, usage):
    csv_file_path = get_csv_file_path(model, classification_type)
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["request", "category", "input_tokens", "output_tokens", "total_tokens"])
        writer.writerow([input_text, classification, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens])
