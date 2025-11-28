import json
import os
from datetime import datetime


def save_response(response: dict, run_name: str):
    os.makedirs("responses", exist_ok=True)
    filename = f"responses/{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(response, f, indent=2)
