# ai/prompt_builder.py
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Example type aliases for clarity (adjust based on your actual data structures)
# e.g., [{'timestamp': '...', 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}, ...]
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
    """
    Build a dynamic, recurring AI prompt for the trading loop.
    This integrates comprehensive live market state data in structured format,
    including timestamp, account details, multi-timeframe price history,
    technical indicators, predictive signals, and a trading rules recap.

    The response is structured as valid JSON for easy parsing.

    Args:
        timestamp: Current timestamp
        minutes_elapsed: Minutes since the bot started
        account_balance: Current account balance
        equity: Current account equity
        open_positions: List of open positions
        price_history_short: Short-term price history (e.g., 10-minute candles)
        price_history_long: Long-term price history (e.g., 40-hour candles)
        indicators: Technical indicators
        predictive_signals: Predictive market signals
        config: Configuration dictionary
        symbol: Optional symbol override
        currency: Optional currency override
    """
    # Use config values with fallbacks
    symbol = symbol or config.get('SYMBOL', 'BTCUSDT')
    currency = currency or config.get('CURRENCY', 'USDT')
    crypto_name = config.get('CRYPTO', 'Bitcoin')
    leverage = config.get('LEVERAGE', 2)
    # Convert to percentage for display
    taker_fee = config.get('TAKER_FEE', 0.0006) * 100

    # Format open positions
    positions_str = "\n".join([
        f"- {pos.get('side','?')} {pos.get('size',0)} {symbol} @${pos.get('entry_price',0):,.0f} PnL${pos.get('unrealized_pnl',0):+.0f}"
        for pos in open_positions
    ]) or "No open positions."

    # Format price history
    def format_ohlcv(history: OHLCV, timeframe: str, limit: int = 10) -> str:
        lines = [f"{timeframe} OHLCV (last {limit}):"]
        for candle in history[-limit:]:
            ts = candle.get('timestamp', 'Unknown')
            o, h, l, c, v = candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
            lines.append(f"- {ts}: O={o:,.1f} H={h:,.1f} L={l:,.1f} C={c:,.1f} V={v:,.0f}")
        return "\n".join(lines)

    # Timeframes from config with fallback values
    cycle_minutes = config.get('CYCLE_MINUTES', 10)
    short_tf = f"{cycle_minutes}-minute"
    long_tf = f"{cycle_minutes * 4}-hour"  # 4x cycle for long-term view

    # Trading rules from config with fallback values
    # Convert percentage to decimal
    max_pos_size_pct = config.get('MAX_POSITION_SIZE_PCT', 0.1) * 100
    daily_loss_pct = config.get('DAILY_LOSS_LIMIT_PCT', 0.02) * 100
    max_drawdown_pct = config.get('MAX_DRAWDOWN_PCT', 0.05) * 100
    max_hold_hours = config.get('MAX_HOLD_HOURS', 24)

    return f"""
You are a professional {crypto_name} trader: grow equity, preserve capital.

Cycle: {cycle_minutes} min.

STATE:
- T: {timestamp.isoformat()}
- Elap: {minutes_elapsed} min
- Bal: ${account_balance:,.0f} {currency}
- Eq: ${equity:,.0f} {currency}
- Pos: {positions_str}

PRICE HISTORY:
{format_ohlcv(price_history_short, short_tf)}
{format_ohlcv(price_history_long, long_tf)}

INDICATORS:
- RSI: {indicators.get('RSI', 'N/A')}
- MACD h:{indicators.get('MACD', {}).get('hist', 'N/A')} s:{indicators.get('MACD', {}).get('signal', 'N/A')}
- EMA20: {indicators.get('EMA_20', 'N/A'):,.1f}
- BB U:{indicators.get('BB_upper', 'N/A'):,.1f} M:{indicators.get('BB_middle', 'N/A'):,.1f} L:{indicators.get('BB_lower', 'N/A'):,.1f}

SIGNALS:
- Vol: {predictive_signals.get('volatility', 'N/A')}
- OB bid:{predictive_signals.get('order_book_depth_bid', 'N/A')} ask:{predictive_signals.get('order_book_depth_ask', 'N/A')}
- Sent: {predictive_signals.get('sentiment_proxy', 'Neutral')}


RULES:
- Risk/trade: 1% eq
- Pos max: {max_pos_size_pct:.0f}% eq
- Daily loss: {daily_loss_pct:.0f}%
- DD max: {max_drawdown_pct:.0f}%
- Lev: {leverage}x
- SL:0.8% TP:2%
- Fee: {taker_fee:.2f}%
- Hold max: {max_hold_hours}h
- High-prob TF align only


TASK: Analyze {short_tf}/{long_tf} align, indic, signals, price action. High-prob only.

Respond ONLY with valid JSON: {{"action": "BUY" | "SELL" | "CLOSE_POSITION" | "HOLD" | "NO_TRADE"}}
""".strip()
