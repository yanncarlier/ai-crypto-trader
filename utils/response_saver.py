# response_saver.py
import logging
import os
import json
from datetime import datetime


def save_response(response: dict, run_name: str):
    """Save AI response to console and logs/ai_{run_name}.log"""
    try:
        logging.info(f"💾 Response: {response}")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "ai.log")
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "run_name": run_name,
                "response": response
            }) + "\n")
    except Exception as e:
        logging.error(f"❌ Could not log response: {e}")
