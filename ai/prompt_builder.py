# ai/prompt_builder.py
import json
from datetime import datetime
from typing import List, Dict, Any

from config.settings import TradingConfig

# Example type aliases for clarity (adjust based on your actual data structures)
OHLCV = List[Dict[str, Any]]  # e.g., [{'timestamp': '...', 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}, ...]

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
    symbol: str = None,
    currency: str = None
) -> str:
    """
    Build a dynamic, recurring AI prompt for the trading loop.
    This integrates comprehensive live market state data in structured format,
    including timestamp, account details, multi-timeframe price history,
    technical indicators, predictive signals, and a trading rules recap.
    
    The response is structured as valid JSON for easy parsing.
    """
    config = TradingConfig()
    
    # Use config values with fallbacks
    symbol = symbol or config.SYMBOL
    currency = currency or config.CURRENCY
    crypto_name = config.CRYPTO
    leverage = config.LEVERAGE
    position_size = config.POSITION_SIZE
    stop_loss_pct = config.STOP_LOSS_PERCENT
    take_profit_pct = config.TAKE_PROFIT_PERCENT
    taker_fee = config.TAKER_FEE * 100  # Convert to percentage for display

    # Format open positions
    positions_str = "\n".join([
        f"- {pos.get('side', 'Unknown')} {pos.get('size', 0)} {symbol} at entry ${pos.get('entry_price', 0):,.2f}, "
        f"current PnL: ${pos.get('unrealized_pnl', 0):+.2f}"
        for pos in open_positions
    ]) or "No open positions."

    # Format price history
    def format_ohlcv(history: OHLCV, timeframe: str, limit: int = 20) -> str:
        lines = [f"{timeframe} timeframe OHLCV (oldest to newest, last {limit} candles):"]
        for candle in history[-limit:]:
            ts = candle.get('timestamp', 'Unknown')
            o, h, l, c, v = candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
            lines.append(f"- {ts}: O={o:,.2f}, H={h:,.2f}, L={l:,.2f}, C={c:,.2f}, V={v:,.0f}")
        return "\n".join(lines)

    # Timeframes from config
    short_tf = f"{config.CYCLE_MINUTES}-minute"
    long_tf = f"{config.CYCLE_MINUTES * 4}-hour"  # 4x cycle for long-term view

    # Trading rules from config
    rules_recap = f"""
Trading Rules Recap:
- Max position size: {config.MAX_POSITION_SIZE_PCT*100:.0f}% of account equity
- Max daily loss: {config.DAILY_LOSS_LIMIT_PCT*100:.0f}% of account
- Max drawdown: {config.MAX_DRAWDOWN_PCT*100:.0f}% before stopping
- Max hold time: {config.MAX_HOLD_HOURS} hours
- Leverage: {leverage}x
- Position size: {position_size}
- Stop loss: {f'{stop_loss_pct}%' if stop_loss_pct else 'Not set'}
- Take profit: {f'{take_profit_pct}%' if take_profit_pct else 'Not set'}
- Taker fee: {taker_fee:.2f}%
- Prioritize capital preservation over aggressive gains
"""

    return f"""
You are a top-level professional {crypto_name} trader focused on multiplying the account while strictly safeguarding capital.

This is a recurring analysis sent every {config.CYCLE_MINUTES} minutes during live trading.

CURRENT MARKET STATE:
- Timestamp: {timestamp.isoformat()}
- Minutes elapsed since trading start: {minutes_elapsed}
- Account balance: ${account_balance:,.2f} {currency}
- Current equity: ${equity:,.2f} {currency}
- Open positions:
{positions_str}

PRICE HISTORY:
{format_ohlcv(price_history_short, short_tf)}
{format_ohlcv(price_history_long, long_tf)}

Technical Indicators:
- RSI (14): {indicators.get('RSI', 'N/A')}
- MACD: hist={indicators.get('MACD', {}).get('hist', 'N/A')}, signal={indicators.get('MACD', {}).get('signal', 'N/A')}
- EMA 20: {indicators.get('EMA_20', 'N/A'):,.2f}
- Bollinger Bands: upper={indicators.get('BB_upper', 'N/A'):,.2f}, middle={indicators.get('BB_middle', 'N/A'):,.2f}, lower={indicators.get('BB_lower', 'N/A'):,.2f}

Predictive Signals:
- Volatility: {predictive_signals.get('volatility', 'N/A')}
- Order Book Depth: bid {predictive_signals.get('order_book_depth_bid', 'N/A')} {currency}, ask {predictive_signals.get('order_book_depth_ask', 'N/A')} {currency}
- Sentiment: {predictive_signals.get('sentiment_proxy', 'Neutral')}

{rules_recap}
Analyze the current market state across multiple timeframes. Identify short-term signals from the {short_tf} data aligned with the longer-term trend from the {long_tf} data.
Consider technical indicators, predictive signals, current positions, and risk rules.

Respond with valid JSON only (no markdown, no extra text, no explanations outside JSON):
{{
  "interpretation": "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Strong Bearish",
  "confidence": 0.0 to 1.0,
  "reasons": "Detailed reasoning incorporating timeframe alignment, key indicators, volume, predictive signals, and risk considerations.",
  "action": "BUY" | "SELL" | "CLOSE_POSITION" | "HOLD" | "NO_TRADE",
  "size_percent_of_equity": 0.0 to {min(100.0, config.MAX_POSITION_SIZE_PCT*100):.1f} (suggested position size as % of equity, 0 if no new trade),
  "stop_loss_price": null | float,
  "take_profit_price": null | float,
  "additional_notes": "Any trailing stop suggestions, alerts, or observations."
}}
""".strip()