# ai/provider.py
import os
import json
import logging
from datetime import datetime
from typing import Literal, Dict, Any, Optional
import httpx
from pydantic import BaseModel, Field

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


import httpx

...

async def send_request(prompt: str, config: Dict[str, Any], api_key: Optional[str] = None) -> AIOutlook:
    """Send request to AI provider using config for settings
    
    Args:
        prompt: The prompt to send to the AI
        config: Configuration dictionary with LLM settings
        api_key: Optional API key (if not in config)
    """
    api_key = api_key or os.getenv("LLM_API_KEY") or config.get('LLM_API_KEY')
    if not api_key:
        return AIOutlook(interpretation="Neutral", reasons="No API key")
        
    provider = config.get('LLM_PROVIDER', 'deepseek').lower()
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown LLM_PROVIDER={provider}")
        
    url = PROVIDER_CONFIG[provider]["url"]
    
    # Use model from config, or provider default if "default"
    model = config.get('LLM_MODEL', 'default')
    if model == "default":
        model = PROVIDER_CONFIG[provider]["default_model"]
        
    temperature = config.get('LLM_TEMPERATURE', 0.2)
    max_tokens = config.get('LLM_MAX_TOKENS', 800)
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
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers=headers, json=payload, timeout=45)
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
    """Save AI response to console only (no file saving)"""
    try:
        logging.info(f"🤖 AI Response: {outlook.interpretation}")
        logging.info(f"   Reasons: {outlook.reasons[:200]}")
    except Exception:
        pass
