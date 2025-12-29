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

    # Open positions (compact)
    positions_str = "\n".join([
        f"- {pos.get('side', '?')} {pos.get('size', 0)} @ ${pos.get('entry_price', 0):,.0f} | PnL ${pos.get('unrealized_pnl', 0):+.0f}"
        for pos in open_positions
    ]) or "No open positions."

    # Compact OHLCV (last 15 candles instead of 20)
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
    long_tf = f"{cycle_minutes * 4}-min"

    # Risk config
    max_pos_pct = config['MAX_POSITION_SIZE_PCT'] * 100
    daily_loss_pct = config['DAILY_LOSS_LIMIT_PCT'] * 100
    drawdown_pct = config['MAX_DRAWDOWN_PCT'] * 100
    max_hold_hours = config['MAX_HOLD_HOURS']

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
{format_ohlcv(price_history_short, short_tf)}
{format_ohlcv(price_history_long, long_tf)}

INDICATORS:
- RSI(14): {indicators.get('RSI', 'N/A')}
- MACD: hist {indicators.get('MACD', {}).get('hist', 'N/A')} | signal {indicators.get('MACD', {}).get('signal', 'N/A')}
- EMA20: {indicators.get('EMA_20', 'N/A'):,.1f}
- BB: U {indicators.get('BB_upper', 'N/A'):,.1f} | M {indicators.get('BB_middle', 'N/A'):,.1f} | L {indicators.get('BB_lower', 'N/A'):,.1f}

SIGNALS:
- Volatility: {predictive_signals.get('volatility', 'N/A')}
- Order Book: bid {predictive_signals.get('order_book_depth_bid', 'N/A')} | ask {predictive_signals.get('order_book_depth_ask', 'N/A')}
- Sentiment: {predictive_signals.get('sentiment_proxy', 'Neutral')} (override with tools if needed)

RULES (strict):
- Max risk/trade: 1% equity
- Max position: {max_pos_pct}% equity
- Daily loss limit: {daily_loss_pct}%
- Max drawdown: {drawdown_pct}%
- Leverage: {leverage}x
- Default SL: ~0.8% | TP: ~2.0%
- Fee: {taker_fee:.2f}%
- Max hold: {max_hold_hours}h
- Volume simulated → ignore trends
- Only enter on high-probability timeframe alignment

TASK:
Analyze {short_tf} vs {long_tf} trend alignment, price action, indicators, order book, sentiment.
Use tools (web/X search) if real-time context adds value.

Respond with valid JSON only — no text outside, no markdown:
{{
  "interpretation": "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Strong Bearish",
  "confidence": 0.0 to 1.0,
  "action": "BUY" | "SELL" | "CLOSE_POSITION" | "HOLD" | "NO_TRADE"
}}
""".strip()
