import pandas as pd
import os
import json

def list_datasets(data_dir: str) -> list:
    """List all dataset folders in the specified data directory."""
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

def load_dataset(folder_path: str) -> dict:
    """Load CSV and JSON files from a dataset folder into a dictionary of Pandas DataFrames."""
    dataset = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv'):
            dataset[file_name] = pd.read_csv(file_path)
        elif file_name.endswith('.json'):
            with open(file_path, 'r') as f:
                dataset[file_name] = json.load(f)
    return dataset