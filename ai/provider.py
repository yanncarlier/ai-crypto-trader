# ai/provider.py
import os
import json
import logging
from datetime import datetime
from typing import Literal, Dict, Any, Optional
from typing import Literal  # already there but ensure
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
    action: Optional[Literal["BUY", "SELL", "CLOSE_POSITION", "HOLD", "NO_TRADE"]] = None
    confidence: Optional[float] = Field(ge=0.0, le=1.0, default=0.5)


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

    temperature = config.get('LLM_TEMPERATURE', 0.3)
    max_tokens = config.get('LLM_MAX_TOKENS', 2000)
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
    logging.getLogger('ai').info(f"=== PROMPT ===\n{prompt}\n=== END PROMPT ===")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers=headers, json=payload, timeout=45)
            r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
        logging.getLogger('ai').info(f"Raw AI content: {content}")
        # Try to parse JSON, handling markdown code fences more robustly
        import re
        # Remove markdown code fences (```json ... ```) possibly with language specifier
        content = re.sub(r'^```[a-z]*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
        content = content.strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Extract fields with fallbacks
                action = parsed.get('action')
                interpretation = parsed.get('interpretation')
                confidence = parsed.get('confidence')
                reasons = parsed.get('reasons', '')
                
                # If interpretation not provided, infer from action if possible
                if not interpretation and action:
                    interp_map = {
                        "BUY": "Bullish",
                        "SELL": "Bearish",
                        "HOLD": "Neutral",
                        "NO_TRADE": "Neutral",
                        "CLOSE_POSITION": "Neutral"
                    }
                    interpretation = interp_map.get(action, "Neutral")
                elif not interpretation:
                    interpretation = "Neutral"
                    
                # Validate interpretation matches allowed values
                if interpretation not in ["Bullish", "Bearish", "Neutral"]:
                    # Map similar strings
                    interpretation_lower = interpretation.lower()
                    if "bull" in interpretation_lower:
                        interpretation = "Bullish"
                    elif "bear" in interpretation_lower:
                        interpretation = "Bearish"
                    else:
                        interpretation = "Neutral"
                
                # Ensure confidence is within bounds
                if confidence is not None:
                    try:
                        confidence = float(confidence)
                        if confidence < 0.0:
                            confidence = 0.0
                        elif confidence > 1.0:
                            confidence = 1.0
                    except (ValueError, TypeError):
                        confidence = 0.5
                else:
                    confidence = 0.5
                    
                # Use provided reasons or default
                if not reasons:
                    reasons = f"AI decision: {action if action else interpretation}"
                    
                return AIOutlook(
                    interpretation=interpretation,
                    reasons=reasons[:500],  # Limit length
                    action=action,
                    confidence=confidence
                )
            # If parsed is not a dict, fall through to keyword detection
            raise ValueError("Parsed JSON is not a dictionary")
        except Exception as parse_error:
            logging.getLogger('ai').warning(f"JSON parsing failed: {parse_error}, falling back to keyword detection")
            # Fallback keyword detection
            text = content.lower()
            if "bullish" in text and "bearish" not in text:
                interp = "Bullish"
            elif "bearish" in text:
                interp = "Bearish"
            else:
                interp = "Neutral"
            return AIOutlook(interpretation=interp, reasons=content[:200], confidence=0.5)
    except Exception as e:
        logging.getLogger('ai').warning(f"AI error: {e}")
        return AIOutlook(interpretation="Neutral", reasons=f"Error: {e}")


def save_response(outlook: AIOutlook, run_name: str) -> None:
    """Save AI response to console only (no file saving)"""
    try:
        logging.getLogger('ai').info(
            f"🤖 AI Response: {outlook.interpretation}")
        logging.getLogger('ai').info(f" : {outlook.reasons[:200]}")
    except Exception:
        pass
