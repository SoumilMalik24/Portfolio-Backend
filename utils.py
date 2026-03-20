import hashlib
import json
import os

METADATA_FILE = "metadata.json"

def get_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def save_metadata(data):
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f)