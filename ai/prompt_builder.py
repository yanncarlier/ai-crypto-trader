def build_prompt(price: float, change_pct: float, volume: str, cycle: int, symbol: str) -> str:
    return f"""
You are a professional Bitcoin trading analyst with access to real-time data.
CURRENT {symbol} PRICE: ${price:,.2f} USDT
{cycle}-minute change: {change_pct:+.2f}%
Volume (USDT): {volume}
Analyze the very short-term momentum and sentiment.
Respond with valid JSON only (no markdown, no extra text):
{{
  "interpretation": "Bullish" | "Bearish" | "Neutral",
  "reasons": "Brief technical + sentiment reasoning using the exact price and recent move."
}}
""".strip()
