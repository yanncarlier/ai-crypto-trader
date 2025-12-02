# ai/prompt_builder_grok.py
def build_prompt(price: float, change_pct: float, volume_24h: float, volume_cycle: float,
                 cycle: int, symbol: str = "BTC") -> str:
    # rough 24h average for this cycle length
    avg_volume_per_cycle = volume_24h / (24 * 60 / cycle)
    volume_ratio = volume_cycle / \
        avg_volume_per_cycle if avg_volume_per_cycle > 0 else 1.0
    return f"""
You are an expert crypto momentum analyst. Your only job is to classify the last {cycle}-minute momentum as Bullish, Bearish, or Neutral.
Data (use exactly these numbers, never round or approximate):
- Current {symbol}/USDT price: ${price:,.2f}
- {cycle}-minute price change: {change_pct:+.2f}%
- {cycle}-minute volume: ${volume_cycle:,.0f} USDT
- Expected average {cycle}-minute volume (from 24h): ${avg_volume_per_cycle:,.0f} USDT (ratio {volume_ratio:.2f}x)
Rules:
- Bullish if clear upward momentum + volume confirmation (ideally ≥1.5x average)
- Bearish if clear downward momentum + volume confirmation
- Neutral if sideways, low volume, or conflicting signals
Answer with valid JSON only. No markdown, no explanations outside the JSON.
{{
  "interpretation": "Bullish" | "Bearish" | "Neutral",
  "reasons": "One-sentence technical reasoning using the exact price, % change, and volume ratio provided above. Max 25 words."
}}
""".strip()
