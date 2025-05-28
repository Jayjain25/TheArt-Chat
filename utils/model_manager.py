import json
import os

MODELS_FILE = "data/custom_models.json"

def ensure_data_directory():
    os.makedirs("data", exist_ok=True)

def load_custom_models():
    ensure_data_directory()
    try:
        if os.path.exists(MODELS_FILE):
            with open(MODELS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading custom models: {e}")
        return {}

def save_custom_models(models_dict):
    ensure_data_directory()
    try:
        with open(MODELS_FILE, 'w') as f:
            json.dump(models_dict, f, indent=4)
    except Exception as e:
        print(f"Error saving custom models: {e}")