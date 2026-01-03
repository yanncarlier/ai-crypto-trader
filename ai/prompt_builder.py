# ai/prompt_builder.py
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

OHLCV = List[Dict[str, Any]]


def build_prompt(
    timestamp: datetime,
    minutes_elapsed: int,
    account_balance: float,
    equity: float,
    open_positions: List[Dict[str, Any]],
    price_history_short: OHLCV,
    price_history_long: OHLCV,
    indicators: Dict[str, Any],
    predictive_signals: Dict[str, Any],
    config: Dict[str, Any],
    symbol: Optional[str] = None,
    currency: Optional[str] = None
) -> str:

    symbol = symbol or config['SYMBOL']
    currency = currency or config['CURRENCY']
    crypto_name = config['CRYPTO']
    leverage = config['LEVERAGE']
    taker_fee = config['TAKER_FEE'] * 100

    # AI prompt configuration
    max_risk_pct = config.get('MAX_RISK_PERCENT', 1.0)
    rsi_period = config.get('RSI_PERIOD', 14)
    ema_period = config.get('EMA_PERIOD', 20)
    bb_period = config.get('BB_PERIOD', 20)
    long_tf_multiplier = config.get('LONG_TF_MULTIPLIER', 4)
    ohlcv_limit = config.get('OHLCV_LIMIT', 15)

    # Open positions (compact)
    positions_str = "\n".join([
        f"- {pos.get('side', '?')} {pos.get('size', 0)} @ ${pos.get('entry_price', 0):,.0f} | PnL ${pos.get('unrealized_pnl', 0):+.0f}"
        for pos in open_positions
    ]) or "No open positions."

    # Compact OHLCV with configurable limit
    def format_ohlcv(history: OHLCV, timeframe: str, limit: int = 15) -> str:
        lines = [f"{timeframe} OHLCV (last {limit}):"]
        for c in history[-limit:]:
            ts = c.get('timestamp', '?')
            lines.append(
                f"- {ts} | O={c['open']:,.1f} H={c['high']:,.1f} L={c['low']:,.1f} C={c['close']:,.1f}"
            )
        return "\n".join(lines)

    cycle_minutes = config['CYCLE_MINUTES']
    short_tf = f"{cycle_minutes}-min"
    long_tf = f"{cycle_minutes * long_tf_multiplier}-min"

    # Risk config

    # Current price comparison
    candle_price = indicators.get('price', 0)
    live_price = indicators.get('live_price', 0)
    price_diff = indicators.get('price_diff', 0)
    price_diff_pct = indicators.get('price_diff_pct', 0)

    return f"""
You are a professional {crypto_name} scalper: grow equity, preserve capital strictly.

Cycle: every {cycle_minutes} min

STATE:
- Time: {timestamp.isoformat()}
- Equity: ${equity:,.0f} {currency}
- Positions: {positions_str}

CURRENT PRICE:
- Candle Close: ${candle_price:,.1f}
- Live Price: ${live_price:,.1f}
- Difference: ${price_diff:+,.1f} ({price_diff_pct:+.2f}%)

PRICE:
{format_ohlcv(price_history_short, short_tf, ohlcv_limit)}
{format_ohlcv(price_history_long, long_tf, ohlcv_limit)}

INDICATORS:
- RSI({rsi_period}): {indicators.get('RSI', 'N/A')}
- MACD: hist {indicators.get('MACD', {}).get('hist', 'N/A')} | signal {indicators.get('MACD', {}).get('signal', 'N/A')}
- EMA{ema_period}: {indicators.get('EMA_20', 'N/A'):,.1f}
- BB({bb_period}): U {indicators.get('BB_upper', 'N/A'):,.1f} | M {indicators.get('BB_middle', 'N/A'):,.1f} | L {indicators.get('BB_lower', 'N/A'):,.1f} | Pos {indicators.get('bb_position', 'N/A'):.2f}
- SMA{ema_period}: {indicators.get('sma_20', 'N/A'):,.1f}
- SMA{ema_period*2}: {indicators.get('sma_50', 'N/A'):,.1f}
- Trend: {indicators.get('trend', 'N/A')}

SIGNALS:
- Volatility: ATR({config.get('ATR_PERIOD', 14)}) = {predictive_signals.get('volatility', 'N/A')}

RULES (strict):
- Max risk/trade: {max_risk_pct:.1f}% equity
- Leverage: {leverage}x
- Fee: {taker_fee:.2f}%
- Enter on strong trend alignment between timeframes
- Exit when AI signals trend reversal
- Preserve capital, avoid over-leveraging

TASK:
Analyze {short_tf} vs {long_tf} trend alignment, price action, indicators, order book, sentiment.
Use tools (web/X search) if real-time context adds value.

Respond with valid JSON only — no text outside, no markdown:
{{
  "interpretation": "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Strong Bearish",
  "confidence": 0.0 to 1.0,
  "action": "OPEN_LONG" | "OPEN_SHORT" | "CLOSE_POSITION" | "HOLD" | "NO_TRADE"
}}
""".strip()
