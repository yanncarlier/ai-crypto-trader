# prompt_builder_gemini.py
import json


def _calculate_volume_velocity(volume_cycle: float, volume_24h: float, cycle_minutes: int) -> str:
    """
    Internal helper to calculate Volume Velocity.
    It compares the current cycle's volume against the 24h average for that specific time duration.
    Returns:
        A string describing the relative volume (e.g., "VERY HIGH (3.2x average)")
    """
    if cycle_minutes <= 0 or volume_24h <= 0:
        return "N/A"
    # Calculate how many cycles fit in a day to find the "average" per cycle
    cycles_per_day = (24 * 60) / cycle_minutes
    avg_vol_per_cycle = volume_24h / cycles_per_day
    # Avoid division by zero
    if avg_vol_per_cycle == 0:
        return "N/A"
    ratio = volume_cycle / avg_vol_per_cycle
    if ratio > 2.5:
        return f"EXTREME ({ratio:.1f}x avg)"
    elif ratio > 1.5:
        return f"HIGH ({ratio:.1f}x avg)"
    elif ratio < 0.6:
        return f"LOW ({ratio:.1f}x avg)"
    else:
        return f"NORMAL ({ratio:.1f}x avg)"


def build_prompt(
    symbol: str,
    cycle_minutes: int,
    current_price: float,
    open_price: float,
    high_price: float,
    low_price: float,
    volume_cycle: float,
    volume_24h: float
) -> str:
    """
    Builds a context-rich prompt for Gemini (or other LLMs) using OHLC data.
    Args:
        symbol: Ticker symbol (e.g., "BTC")
        cycle_minutes: The timeframe in minutes (e.g., 5, 15, 60)
        current_price: The current live price
        open_price: The price at the start of this cycle
        high_price: The highest price reached in this cycle
        low_price: The lowest price reached in this cycle
        volume_cycle: The volume traded strictly within this cycle
        volume_24h: The rolling 24h volume
    Returns:
        A formatted string prompt ready to be sent to the LLM.
    """
    # 1. Calculate Technical Metrics
    pct_change = ((current_price - open_price) / open_price) * 100
    range_pct = ((high_price - low_price) / low_price) * 100
    vol_context = _calculate_volume_velocity(
        volume_cycle, volume_24h, cycle_minutes)
    # 2. Determine Position in Range (Where are we closing?)
    # 0.0 = Low, 1.0 = High
    try:
        position_in_range = (current_price - low_price) / \
            (high_price - low_price)
    except ZeroDivisionError:
        position_in_range = 0.5  # Flat candle
    range_desc = "Mid-range"
    if position_in_range > 0.8:
        range_desc = "Near Highs (Strong Close)"
    elif position_in_range < 0.2:
        range_desc = "Near Lows (Weak Close)"
    # 3. Construct the Prompt
    return f"""
You are a professional Algorithmic Trading Analyst for {symbol}.
Your goal is to analyze the momentum of the current {cycle_minutes}-minute candle.
### MARKET DATA ({cycle_minutes}-minute timeframe)
- **Symbol:** {symbol}/USDT
- **Current Price:** ${current_price:,.2f}
- **Cycle Open:** ${open_price:,.2f}
- **Cycle High:** ${high_price:,.2f} (Immediate Resistance)
- **Cycle Low:** ${low_price:,.2f} (Immediate Support)
- **Price Change:** {pct_change:+.2f}%
- **Candle Range:** {range_desc} (Volatility: {range_pct:.2f}%)
### VOLUME ANALYSIS
- **Cycle Volume:** ${volume_cycle:,.0f}
- **24h Volume:** ${volume_24h:,.0f}
- **Volume Velocity:** {vol_context}
(Note: Volume Velocity compares current activity to the 24h average. >1.5x is significant.)
### ANALYTICAL TASKS
1. **Momentum:** Does the Close vs High/Low suggest continuation or rejection?
2. **Conviction:** Does the "Volume Velocity" support the price move? (e.g., High volume breakout is bullish; Low volume breakout is suspicious).
3. **Sentiment:** Combine price action + volume to determine short-term sentiment.
### RESPONSE FORMAT
Respond with VALID JSON ONLY. Do not use Markdown code blocks. Do not add conversational text.
{{
  "signal": "BULLISH_MOMENTUM" | "BEARISH_MOMENTUM" | "CONSOLIDATION" | "INDECISION",
  "confidence": 1-10,
  "analysis": "Brief, dense technical reasoning focusing on Volume Velocity ({vol_context}) and Price Location ({range_desc}).",
  "trigger_alert": true | false
}}
""".strip()


# --- Execution Block for Testing ---
if __name__ == "__main__":
    # Test Data: Simulating a fake "Pump" scenario
    # Price moved up 1.2% in 15 mins with 3x normal volume
    test_data = {
        "symbol": "BTC",
        "cycle_minutes": 15,
        "current_price": 96150.00,
        "open_price": 95000.00,
        "high_price": 96200.00,
        "low_price": 94800.00,
        "volume_cycle": 50_000_000.0,
        "volume_24h": 1_500_000_000.0
    }
    print("--- Generating Prompt ---")
    prompt = build_prompt(
        symbol=test_data["symbol"],
        cycle_minutes=test_data["cycle_minutes"],
        current_price=test_data["current_price"],
        open_price=test_data["open_price"],
        high_price=test_data["high_price"],
        low_price=test_data["low_price"],
        volume_cycle=test_data["volume_cycle"],
        volume_24h=test_data["volume_24h"]
    )
    print(prompt)
