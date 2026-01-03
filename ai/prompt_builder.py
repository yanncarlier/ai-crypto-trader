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
    min_rr_ratio = config.get('MIN_RISK_REWARD_RATIO', 2.0)
    confidence_threshold = config.get('CONFIDENCE_THRESHOLD', 0.7)
    weekly_growth_target = config.get('WEEKLY_GROWTH_TARGET', 5.0)
    rsi_period = config.get('RSI_PERIOD', 14)
    ema_period = config.get('EMA_PERIOD', 20)
    bb_period = config.get('BB_PERIOD', 20)
    long_tf_multiplier = config.get('LONG_TF_MULTIPLIER', 4)
    ohlcv_limit = config.get('OHLCV_LIMIT', 7)

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
You are an aggressive {crypto_name} momentum trader: compound equity growth through decisive trend following.

TARGET: {weekly_growth_target:.1f}% weekly growth through consistent 2-3x risk-reward trades.

STATE:
- Equity: ${equity:,.0f} {currency}
- Positions: {positions_str}

TREND ANALYSIS:
{format_ohlcv(price_history_short, short_tf, ohlcv_limit)}
{format_ohlcv(price_history_long, long_tf, ohlcv_limit)}

KEY INDICATORS:
- Trend: {indicators.get('trend', 'N/A')}
- RSI({rsi_period}): {indicators.get('RSI', 'N/A')}
- BB Position: {indicators.get('bb_position', 'N/A'):.2f}

RULES (aggressive):
- Risk/Reward: Minimum {min_rr_ratio:.1f}x ratio per trade
- Max risk/trade: {max_risk_pct:.1f}% equity
- Leverage: {leverage}x optimized for growth
- Confidence threshold: {confidence_threshold:.1f}+ required
- Reinvest ALL profits to compound growth
- Enter strong trends immediately, exit on reversal signals

TASK:
Identify trend direction and momentum strength. Execute high-probability trades with {min_rr_ratio:.1f}x+ targets.

Respond with valid JSON only:
{{
  "interpretation": "STRONG_UPTREND" | "STRONG_DOWNTREND",
  "confidence": {confidence_threshold:.1f} to 1.0,
  "action": "OPEN_LONG" | "OPEN_SHORT" | "CLOSE_POSITION"
}}
""".strip()
