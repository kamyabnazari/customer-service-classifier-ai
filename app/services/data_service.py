import pandas as pd
import os
import json

def list_datasets(data_dir: str) -> list:
    """List all dataset folders in the specified data directory."""
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

def load_dataset(folder_path: str) -> dict:
    """Load CSV files from a dataset folder into a dictionary of Pandas DataFrames."""
    dataset = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv'):
            dataset[file_name] = pd.read_csv(file_path)
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