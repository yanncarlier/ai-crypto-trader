# response_saver.py
import json
import os
import logging
from datetime import datetime
from pathlib import Path


def save_response(response: dict, run_name: str):
    """Legacy function - now uses the combined logs directory"""
    try:
        # Use the new logs directory structure
        path = Path("logs/ai_responses")
        path.mkdir(parents=True, exist_ok=True)
        filename = f"logs/ai_responses/{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(response, f, indent=2)
        logging.info(f"💾 Legacy response saved to: {filename}")
    except Exception as e:
        logging.error(f"❌ Could not save legacy response: {e}")
