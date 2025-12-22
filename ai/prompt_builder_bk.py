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
    position_size = config.get('POSITION_SIZE', '10%')
    stop_loss_pct = config.get('STOP_LOSS_PERCENT', 10)
    take_profit_pct = config.get('TAKE_PROFIT_PERCENT')
    # Convert to percentage for display
    taker_fee = config.get('TAKER_FEE', 0.0006) * 100

    # Format open positions
    positions_str = "\n".join([
        f"- {pos.get('side', 'Unknown')} {pos.get('size', 0)} {symbol} at entry ${pos.get('entry_price', 0):,.2f}, "
        f"current PnL: ${pos.get('unrealized_pnl', 0):+.2f}"
        for pos in open_positions
    ]) or "No open positions."

    # Format price history
    def format_ohlcv(history: OHLCV, timeframe: str, limit: int = 20) -> str:
        lines = [
            f"{timeframe} timeframe OHLCV (oldest to newest, last {limit} candles):"]
        for candle in history[-limit:]:
            ts = candle.get('timestamp', 'Unknown')
            o, h, l, c, v = candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
            lines.append(
                f"- {ts}: O={o:,.2f}, H={h:,.2f}, L={l:,.2f}, C={c:,.2f}, V={v:,.0f}")
        return "\n".join(lines)

    # Timeframes from config with fallback values
    cycle_minutes = config.get('CYCLE_MINUTES', 10)
    short_tf = f"{cycle_minutes}-minute"
    long_tf = f"{cycle_minutes * 4}-hour"  # 4x cycle for long-term view

    # Trading rules from config with fallback values
    # Convert percentage to decimal
    max_pos_size_pct = float(config.get('MAX_POSITION_SIZE_PCT', 10)) / 100
    daily_loss_pct = float(config.get('DAILY_LOSS_LIMIT_PCT', 2)) / 100
    max_drawdown_pct = float(config.get('MAX_DRAWDOWN_PCT', 5)) / 100
    max_hold_hours = int(config.get('MAX_HOLD_HOURS', 24))

    rules_recap = f"""
Trading Rules Recap:
- Max position size: {max_pos_size_pct*100:.0f}% of account equity
- Max daily loss: {daily_loss_pct*100:.0f}% of account
- Max drawdown: {max_drawdown_pct*100:.0f}% before stopping
- Max hold time: {max_hold_hours} hours
- Leverage: {leverage}x
- Position size: {position_size}
- Stop loss: {f'{stop_loss_pct}%' if stop_loss_pct else 'Not set'}
- Take profit: {f'{take_profit_pct}%' if take_profit_pct else 'Not set'}
- Taker fee: {taker_fee:.2f}%
- Prioritize capital preservation — only take high-probability setups aligned across timeframes
- Volume data appears simulated/placeholder — do not base primary decisions on volume trends
"""

    return f"""
You are a top-level professional {crypto_name} trader focused on multiplying the account while strictly safeguarding capital through disciplined risk management.

This analysis runs every {cycle_minutes} minute(s) during live trading.

CURRENT MARKET STATE:
- Timestamp: {timestamp.isoformat()}
- Minutes elapsed since trading start:  {minutes_elapsed}
- Account balance: ${account_balance:,.2f} {currency}
- Current equity: ${equity:,.2f} {currency}
- Open positions: {positions_str}

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


Task:
Analyze the current market across multiple timeframes. Identify short-term signals from the {short_tf} chart that align with the longer-term trend from the {long_tf} chart.
Consider technical indicators, price action, order book imbalance, volatility, and sentiment.
If using tools (e.g., web/X search), feel free to fetch real-time sentiment or news for Bitcoin.

Respond with valid JSON only — no markdown, no extra text, no explanations outside the JSON:
{{
    "interpretation": "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Strong Bearish",
    "confidence": 0.0 to 1.0,
    "reasons": "Concise reasoning (max 120 words): timeframe alignment, key indicators, price action, risk considerations.",
    "action": "BUY" | "SELL" | "CLOSE_POSITION" | "HOLD" | "NO_TRADE",
    "size_percent_of_equity": 0.0 to  {min(100.0, max_pos_size_pct*100):.1f}  (0.0 if no new trade; suggest 0.1 –  {min(100.0, max_pos_size_pct*100):.1f}  only on high-confidence aligned setups),
    "stop_loss_price": null 
    "take_profit_price": null 
    "additional_notes": "Brief notes: trailing stop ideas, alerts, or key levels to watch."
}}
""".strip()
