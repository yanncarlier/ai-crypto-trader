# ai/prompt_builder.py
from config.settings import TradingConfig

def build_prompt(price: float, change_pct: float, volume_24h: float, volume_cycle: float,
                 cycle: int, symbol: str) -> str:
    """Build AI prompt with dynamic cycle minutes from config"""
    config = TradingConfig()
    return f"""
You are a top-level professional {config.CRYPTO} trader focused on multiplying the account while safeguarding capital, with access to real-time data.
CURRENT {symbol} PRICE: ${price:,.2f} {config.CURRENCY}
{cycle}-minute price change: {change_pct:+.2f}%
24h Volume: ${volume_24h:,.0f} {config.CURRENCY}
Last {cycle}-minute Volume: ${volume_cycle:,.0f} {config.CURRENCY}
Current market context:
- Trading cycle: {cycle} minutes
- Momentum analysis requested
Analyze the very Momentum and market sentiment based on the {cycle}-minute timeframe.
Is the price showing strength, weakness, or consolidation in the recent {cycle} minutes?
Key considerations:
1. {cycle}-minute price change of {change_pct:+.2f}%
2. Volume activity in last {cycle} minutes vs 24h average
3. Short-term trend direction and momentum
4. Recent support/resistance levels (implied by price action)
Respond with valid JSON only (no markdown, no extra text):
{{
  "interpretation": "Bullish" | "Bearish" | "Neutral",
  "reasons": "Brief technical + sentiment reasoning focusing on {cycle}-minute timeframe, using exact price ${price:,.2f}, {change_pct:+.2f}% change, and volume activity (${volume_cycle:,.0f} in {cycle}min vs ${volume_24h:,.0f} 24h)."
}}
""".strip()
