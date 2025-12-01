# ai/provider.py
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Literal
import requests
from pydantic import BaseModel, Field
from config.settings import TradingConfig
# Provider configuration
PROVIDER_CONFIG = {
    "xai": {"url": "https://api.x.ai/v1/chat/completions", "default_model": "grok-2-1212"},
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "default_model": "llama-3.3-70b-versatile"},
    "openai": {"url": "https://api.openai.com/v1/chat/completions", "default_model": "gpt-4o-mini"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "default_model": "google/gemini-flash-1.5"},
    "deepseek": {"url": "https://api.deepseek.com/beta/chat/completions", "default_model": "deepseek-chat"},
    "mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "default_model": "mistral-large-latest"},
}


class AIOutlook(BaseModel):
    interpretation: Literal["Bullish", "Bearish", "Neutral"]
    reasons: str = Field(min_length=1)


def send_request(prompt: str, config: TradingConfig, api_key: str | None = None) -> AIOutlook:
    """Send request to AI provider using config for settings"""
    api_key = api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        return AIOutlook(interpretation="Neutral", reasons="No API key")
    provider = config.LLM_PROVIDER.lower()
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown LLM_PROVIDER={provider}")
    url = PROVIDER_CONFIG[provider]["url"]
    # Use model from config, or provider default if "default"
    if config.LLM_MODEL == "default":
        model = PROVIDER_CONFIG[provider]["default_model"]
    else:
        model = config.LLM_MODEL
    temperature = config.LLM_TEMPERATURE
    max_tokens = config.LLM_MAX_TOKENS
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
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
