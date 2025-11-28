# ai/provider.py
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Literal
import requests
from pydantic import BaseModel, Field
# ===================== CONFIG =====================
PROVIDER = os.getenv("LLM_PROVIDER", "xai").lower()
PROVIDER_CONFIG = {
    "xai":      {"url": "https://api.x.ai/v1/chat/completions",           "default_model": "grok-2-1212"},
    "groq":     {"url": "https://api.groq.com/openai/v1/chat/completions", "default_model": "llama-3.3-70b-versatile"},
    "openai":   {"url": "https://api.openai.com/v1/chat/completions",     "default_model": "gpt-4o-mini"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "default_model": "google/gemini-flash-1.5"},
    "deepseek": {"url": "https://api.deepseek.com/beta/chat/completions", "default_model": "deepseek-chat"},
    "mistral":  {"url": "https://api.mistral.ai/v1/chat/completions",    "default_model": "mistral-large-latest"},
}
if PROVIDER not in PROVIDER_CONFIG:
    raise ValueError(f"Unknown LLM_PROVIDER={PROVIDER}")
URL = PROVIDER_CONFIG[PROVIDER]["url"]
MODEL = os.getenv("LLM_MODEL", PROVIDER_CONFIG[PROVIDER]["default_model"])
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
# ===================== MODELS =====================


class AIOutlook(BaseModel):
    interpretation: Literal["Bullish", "Bearish", "Neutral"]
    reasons: str = Field(min_length=1)
# ===================== MAIN FUNCTION =====================


def send_request(prompt: str, crypto_symbol: str = "Bitcoin", api_key: str | None = None) -> AIOutlook:
    api_key = api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        return AIOutlook(interpretation="Neutral", reasons="No API key")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "submit_outlook",
                "description": "Submit market outlook",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interpretation": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
                        "reasons": {"type": "string"}
                    },
                    "required": ["interpretation", "reasons"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }],
        "tool_choice": "auto"
    }
    try:
        logging.debug(f"Calling {PROVIDER.upper()} → {MODEL}")
        r = requests.post(URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        # Tool call (works perfectly with xAI/Grok)
        if msg.get("tool_calls"):
            args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
            return AIOutlook(**args)
        # Direct JSON response
        content = msg.get("content", "").strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            return AIOutlook(**json.loads(content))
        except:
            pass
        # Fallback keyword
        text = content.lower()
        interp = "Bullish" if "bullish" in text and "bearish" not in text else "Bearish" if "bearish" in text else "Neutral"
        return AIOutlook(interpretation=interp, reasons=text[:500])
    except Exception as e:
        logging.error(f"AI call failed: {e}")
        return AIOutlook(interpretation="Neutral", reasons=f"Error: {e}")
# ===================== SAVE RESPONSE =====================


def save_response(outlook: AIOutlook, run_name: str) -> None:
    try:
        path = Path("ai_responses")
        path.mkdir(exist_ok=True)
        file = path / f"{run_name}.json"
        data = {}
        if file.exists():
            data = json.load(open(file))
        data[datetime.utcnow().isoformat()] = outlook.model_dump()
        json.dump(data, open(file, "w"), indent=2)
    except Exception as e:
        logging.error(f"Could not save AI response: {e}")
