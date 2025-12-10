import json
import os
from datetime import datetime


def load_json(path: str):
    """Load JSON file safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    """Save JSON file safely."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def timestamp():
    """Return a formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_banner(text: str):
    """Nice banner for CLI display."""
    print("\n" + "=" * 40)
    print(text)
    print("=" * 40 + "\n")
