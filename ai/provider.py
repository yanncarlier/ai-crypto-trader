# ai/provider.py
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal
import requests
from pydantic import BaseModel, Field
# ===================== PROVIDER CONFIGURATION =====================
# ← DEFAULT IS XAI!
PROVIDER = os.getenv("LLM_PROVIDER", "xai").lower()
PROVIDER_CONFIG = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.3-70b-versatile"
    },
    "xai": {
        "url": "https://api.x.ai/v1/chat/completions",
        "default_model": "grok-2-1212"
    },
    "deepseek": {
        "url": "https://api.deepseek.com/beta/chat/completions",
        "default_model": "deepseek-chat"
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "google/gemini-2.0-flash-exp"
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "default_model": "gemini-1.Flash",
        "auth_header": "key"
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-3-5-sonnet-20241022",
        "anthropic_version": "2023-06-01"
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini"
    },
}
if PROVIDER not in PROVIDER_CONFIG:
    raise ValueError(
        f"Invalid LLM_PROVIDER={PROVIDER}. Choose from: {', '.join(PROVIDER_CONFIG)}")
# FIXED LINE ← this was broken before
URL = PROVIDER_CONFIG[PROVIDER]["url"]
MODEL = os.getenv("LLM_MODEL", PROVIDER_CONFIG[PROVIDER]["default_model"])
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))


class AIResponseError(Exception):
    pass


class AIOutlook(BaseModel):
    interpretation: Literal["Bullish", "Bearish", "Neutral"]
    reasons: str = Field(min_length=1)


def send_request(prompt: str, crypto_symbol: str = "Bitcoin", api_key: str | None = None) -> AIOutlook:
    api_key = api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        logging.warning("No LLM_API_KEY → returning Neutral")
        return AIOutlook(interpretation="Neutral", reasons="Missing API key")
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "user", "content": prompt}]
    # ===================== GEMINI SPECIAL CASE =====================
    if PROVIDER == "gemini":
        url = f"{URL}?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAX_TOKENS}
        }
        try:
            r = requests.post(url, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            content = data["candidates"][0]["content"]["parts"][0]
            if "functionCall" in content:
                return AIOutlook(**content["functionCall"]["args"])
            return _parse_fallback(content.get("text", ""))
        except Exception as e:
            logging.warning(f"Gemini failed: {e}")
            return _parse_fallback("")
    # ===================== ANTHROPIC SPECIAL CASE =====================
    elif PROVIDER == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = PROVIDER_CONFIG[PROVIDER]["anthropic_version"]
        payload = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": messages,
        }
        # Add tool if you want (optional)
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            for block in data.get("content", []):
                if block.get("type") == "tool_use" and block["name"] == f"{crypto_symbol.lower()}_outlook":
                    return AIOutlook(**block["input"])
            return _parse_fallback(data.get("content", [{}])[0].get("text", ""))
        except Exception as e:
            logging.warning(f"Anthropic error: {e}")
            return AIOutlook(interpretation="Neutral", reasons="Anthropic failed")
    # ===================== STANDARD OPENAI-COMPATIBLE (including xAI) =====================
    else:
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": messages
        }
        # Use tools only on reliable providers (xAI works great with tools!)
        if PROVIDER in {"xai", "openai", "openrouter", "deepseek", "mistral"}:
            payload["tools"] = _openai_tools_schema(crypto_symbol)
            payload["tool_choice"] = "auto"
        logging.info(f"→ {PROVIDER.upper()} ({MODEL})")
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            # Tool call path
            if msg.get("tool_calls"):
                args = json.loads(msg["tool_calls"][0]
                                  ["function"]["arguments"])
                return AIOutlook(**args)
            # Direct JSON or text
            content = msg.get("content", "").strip()
            if not content:
                raise AIResponseError("Empty response")
            try:
                return AIOutlook(**json.loads(content))
            except json.JSONDecodeError:
                return _parse_fallback(content)
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return AIOutlook(interpretation="Neutral", reasons=f"API error: {e}")
# ===================== TOOL SCHEMA =====================


def _openai_tools_schema(crypto_symbol: str):
    return [{
        "type": "function",
        "function": {
            "name": f"{crypto_symbol.lower()}_outlook",
            "description": "Structured market outlook",
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
    }]
# ===================== FALLBACK =====================


def _parse_fallback(text: str) -> AIOutlook:
    text = text.lower()
    if "bullish" in text and "bearish" not in text:
        interp = "Bullish"
    elif "bearish" in text:
        interp = "Bearish"
    else:
        interp = "Neutral"
    reasons = text.strip().split("\n")[0][:500]
    logging.info(f"Fallback parsing → {interp}")
    return AIOutlook(interpretation=interp, reasons=reasons or "No clear signal")
# ===================== SAVE RESPONSE =====================


def save_response(outlook: AIOutlook, run_name: str) -> None:
    try:
        dir_path = Path("ai_responses")
        dir_path.mkdir(exist_ok=True)
        file_path = dir_path / f"{run_name}.json"
        data = {}
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
        ts = datetime.now(timezone.utc).isoformat()
        data[ts] = outlook.model_dump()
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"AI response saved → {file_path}")
    except Exception as e:
        logging.error(f"Failed to save AI response: {e}")
