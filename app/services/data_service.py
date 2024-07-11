import pandas as pd
import os
import csv
import json

base_results_dir = './customer_service_classifier_ai_data/results'

def list_datasets(data_dir: str) -> list:
    """List all dataset folders in the specified data directory, ignoring folders ending with '_generated'."""
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.endswith('_generated')]

def load_dataset(folder_path: str) -> dict:
    """Load CSV files from a dataset folder into a dictionary of Pandas DataFrames."""
    dataset = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
            dataset[file_name] = df
    return dataset

def load_datasets(selected_dataset, data_dir):
    dataset_path = os.path.join(data_dir, selected_dataset)
    dataset = load_dataset(dataset_path)
    
    datasets = {}
    for file_name, data in dataset.items():
        if file_name == 'test.csv':
            datasets["test"] = pd.DataFrame(data)
        elif file_name == 'fine_tuning.csv':
            datasets["fine_tuning"] = pd.DataFrame(data)
        elif file_name == 'categories.csv':
            datasets["categories"] = pd.DataFrame(data)
    return datasets

def csv_to_jsonl(csv_file_path, jsonl_file_path, system_message):
    with open(csv_file_path, mode='r') as csv_file, open(jsonl_file_path, mode='w') as jsonl_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            jsonl_obj = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": row["text"]},
                    {"role": "assistant", "content": row["category"]}
                ]
            }
            jsonl_file.write(json.dumps(jsonl_obj) + "\n")

def list_jsonl_files(directory):
    """List all JSONL files in the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith('.jsonl')]

def get_csv_file_path(model, classification_type, classification_method, temperature):
    results_dir = os.path.join(base_results_dir, classification_method, model)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    csv_file_name = f'classification_results_{model}_{classification_type}_{temperature}.csv'
    return os.path.join(results_dir, csv_file_name)

def check_results_exist(model, classification_type, temperature, classification_method):
    csv_file_path = get_csv_file_path(model, classification_type, classification_method, temperature)
    return os.path.isfile(csv_file_path)