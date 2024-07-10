import os
from datetime import datetime, timezone

# Basis-Log-Verzeichnis
base_log_dir = './customer_service_classifier_ai_data/logs'

def get_log_file_path(model, classification_type):
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    log_dir = os.path.join(base_log_dir, f'{model}_{classification_type}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = f'classification_logs_{current_date}.txt'
    return os.path.join(log_dir, log_file_name)

def write_log_to_file(model, classification_type, system_message, input_text, classification):
    log_file_path = get_log_file_path(model, classification_type)
    log_message = (
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"Model: {model}\n"
        f"Classification Type: {classification_type}\n"
        f"System Message: {system_message}\n"
        f"Input Text: {input_text}\n"
        f"Classification: {classification}\n"
        "----------------------------------------\n"
    )
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_message)