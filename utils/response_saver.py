# response_saver.py
import logging


def save_response(response: dict, run_name: str):
    """Legacy function - now logs to console only"""
    try:
        logging.info(f"💾 Response: {response}")
    except Exception as e:
        logging.error(f"❌ Could not log response: {e}")
