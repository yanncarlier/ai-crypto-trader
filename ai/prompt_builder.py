# ai/prompt_builder.py
def build_prompt(price: float, change_pct: float, volume: float, cycle: int, symbol: str) -> str:
    return f"""
You are a professional Bitcoin trading analyst with access to real-time data.
CURRENT {symbol} PRICE: ${price:,.2f} USDT
{cycle}-minute price change: {change_pct:+.2f}%
24h Volume: ${volume:,.0f} USDT
Analyze the very short-term momentum and market sentiment.
Is the price showing strength, weakness, or consolidation?
Respond with valid JSON only (no markdown, no extra text):
{{
  "interpretation": "Bullish" | "Bearish" | "Neutral",
  "reasons": "Brief technical + sentiment reasoning using the exact price and recent move."
}}
""".strip()
