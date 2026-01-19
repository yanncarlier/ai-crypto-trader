import os
import json
import logging
import re
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
    interpretation: Literal["STRONG_UPTREND", "STRONG_DOWNTREND"]
    reasons: str = Field(min_length=1)
    action: Optional[Literal["OPEN_LONG", "OPEN_SHORT", "CLOSE_POSITION"]] = None
    confidence: Optional[float] = Field(ge=0.0, le=1.0, default=0.5)


async def send_request(prompt: str, config: Dict[str, Any], api_key: Optional[str] = None) -> AIOutlook:
    """Send request to AI provider using config for settings and return a validated AIOutlook.

    Args:
        prompt: The prompt to send to the AI
        config: Configuration dictionary with LLM settings
        api_key: Optional API key (if not in config)
    """
    api_key = api_key or os.getenv("LLM_API_KEY") or config.get("LLM_API_KEY")
    if not api_key:
        # Default to uptrend with medium confidence to keep pipeline stable
        return AIOutlook(
            interpretation="STRONG_UPTREND",
            reasons="No API key provided",
            action=None,
            confidence=0.5,
        )

    provider = config.get("LLM_PROVIDER", "deepseek").lower()
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown LLM_PROVIDER={provider}")

    url = PROVIDER_CONFIG[provider]["url"]

    # Use model from config, or provider default if "default"
    model = config.get("LLM_MODEL", "default")
    if model == "default":
        model = PROVIDER_CONFIG[provider]["default_model"]

    temperature = float(config.get("LLM_TEMPERATURE", 0.3))
    max_tokens = int(config.get("LLM_MAX_TOKENS", 2000))
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    logging.getLogger("ai").info(f"=== PROMPT ===\n{prompt}\n=== END PROMPT ===")

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers=headers, json=payload, timeout=45)
            r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
        logging.getLogger("ai").info(f"Raw AI content: {content}")

        # Remove markdown code fences (```json ... ```) possibly with language specifier
        content = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n```$", "", content, flags=re.MULTILINE).strip()

        # Try to parse JSON
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON is not a dictionary")

            action = parsed.get("action")
            interpretation = parsed.get("interpretation")
            confidence = parsed.get("confidence")
            reasons = parsed.get("reasons", "")

            # Validate/normalize action
            if action not in ("OPEN_LONG", "OPEN_SHORT", "CLOSE_POSITION"):
                action = None

            # If interpretation not provided, infer from action; else default
            if not interpretation and action:
                interp_map = {
                    "OPEN_LONG": "STRONG_UPTREND",
                    "OPEN_SHORT": "STRONG_DOWNTREND",
                    "CLOSE_POSITION": "STRONG_UPTREND",
                }
                interpretation = interp_map.get(action, "STRONG_UPTREND")
            elif not interpretation:
                interpretation = "STRONG_UPTREND"

            # Coerce interpretation to allowed values
            if interpretation not in ("STRONG_UPTREND", "STRONG_DOWNTREND"):
                low = str(interpretation).lower()
                if any(k in low for k in ("up", "bull", "long")):
                    interpretation = "STRONG_UPTREND"
                elif any(k in low for k in ("down", "bear", "short")):
                    interpretation = "STRONG_DOWNTREND"
                else:
                    interpretation = "STRONG_UPTREND"

            # Clamp confidence to [0, 1]
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5

            if not reasons:
                reasons = f"AI decision: {action if action else interpretation}"

            return AIOutlook(
                interpretation=interpretation,
                reasons=reasons[:500],
                action=action,
                confidence=confidence,
            )
        except Exception as parse_error:
            logging.getLogger("ai").warning(
                f"JSON parsing failed: {parse_error}, falling back to keyword detection"
            )
            text = content.lower()
            if any(k in text for k in ("bullish", "up", "long")):
                interp = "STRONG_UPTREND"
            elif any(k in text for k in ("bearish", "down", "short")):
                interp = "STRONG_DOWNTREND"
            else:
                interp = "STRONG_UPTREND"
            return AIOutlook(interpretation=interp, reasons=content[:200], confidence=0.5)

    except Exception as e:
        logging.getLogger("ai").warning(f"AI error: {e}")
        return AIOutlook(interpretation="STRONG_UPTREND", reasons=f"Error: {e}")


def save_response(outlook: AIOutlook, run_name: str) -> None:
    """Save AI response to console only (no file saving)"""
    try:
        logging.getLogger("ai").info(f"🤖 AI Response: {outlook.interpretation}")
        logging.getLogger("ai").info(f" : {outlook.reasons[:200]}")
    except Exception:
        pass
