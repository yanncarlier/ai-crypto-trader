# ai/provider.py
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Literal
import requests
from pydantic import BaseModel, Field
PROVIDER = os.getenv("LLM_PROVIDER", "xai").lower()
PROVIDER_CONFIG = {
    "xai": {"url": "https://api.x.ai/v1/chat/completions", "default_model": "grok-2-1212"},
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "default_model": "llama-3.3-70b-versatile"},
    "openai": {"url": "https://api.openai.com/v1/chat/completions", "default_model": "gpt-4o-mini"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "default_model": "google/gemini-flash-1.5"},
    "deepseek": {"url": "https://api.deepseek.com/beta/chat/completions", "default_model": "deepseek-chat"},
    "mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "default_model": "mistral-large-latest"},
}
if PROVIDER not in PROVIDER_CONFIG:
    raise ValueError(f"Unknown LLM_PROVIDER={PROVIDER}")
URL = PROVIDER_CONFIG[PROVIDER]["url"]
MODEL = os.getenv("LLM_MODEL", PROVIDER_CONFIG[PROVIDER]["default_model"])
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))


class AIOutlook(BaseModel):
    interpretation: Literal["Bullish", "Bearish", "Neutral"]
    reasons: str = Field(min_length=1)


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
    }
    try:
        r = requests.post(URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Try to parse JSON
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            parsed = json.loads(content)
            return AIOutlook(**parsed)
        except:
            # Fallback keyword detection
            text = content.lower()
            if "bullish" in text and "bearish" not in text:
                interp = "Bullish"
            elif "bearish" in text:
                interp = "Bearish"
            else:
                interp = "Neutral"
            return AIOutlook(interpretation=interp, reasons=content[:200])
    except Exception as e:
        logging.warning(f"AI error: {e}")
        return AIOutlook(interpretation="Neutral", reasons=f"Error: {e}")


def save_response(outlook: AIOutlook, run_name: str) -> None:
    try:
        path = Path("logs/ai_responses")
        path.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d')
        file = path / f"{run_name}_{date_str}.json"
        data = {}
        if file.exists():
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
            except:
                pass
        timestamp = datetime.utcnow().isoformat()
        data[timestamp] = outlook.model_dump()
        with open(file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
