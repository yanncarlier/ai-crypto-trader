import asyncio
from datetime import datetime

import pytest

from ai.prompt_builder import build_prompt
from ai.provider import AIOutlook


def test_build_prompt_minimal_config():
    config = {
        'SYMBOL': 'BTCUSDT',
        'CURRENCY': 'USDT',
        'CRYPTO': 'Bitcoin',
        'LEVERAGE': 5,
        'TAKER_FEE': 0.0006,
        'CYCLE_MINUTES': 5,
    }

    prompt = build_prompt(
        timestamp=datetime.utcnow(),
        minutes_elapsed=10,
        account_balance=1000.0,
        equity=1000.0,
        open_positions=[],
        price_history_short=[{'timestamp': 't', 'open':1,'high':2,'low':0.5,'close':1.5,'volume':1}],
        price_history_long=[{'timestamp': 't', 'open':1,'high':2,'low':0.5,'close':1.5,'volume':1}],
        indicators={'trend': 'bullish', 'RSI': 55, 'bb_position': 0.5, 'price': 1.5, 'live_price': 1.5, 'price_diff': 0, 'price_diff_pct': 0},
        predictive_signals={},
        config=config,
    )

    assert isinstance(prompt, str)
    assert 'Bitcoin' in prompt
    assert 'RSI(14)' in prompt
    assert 'OPEN_LONG' in prompt and 'OPEN_SHORT' in prompt and 'CLOSE_POSITION' in prompt


def test_ai_outlook_validation_defaults():
    o = AIOutlook(interpretation="STRONG_UPTREND", reasons="test")
    assert o.interpretation == "STRONG_UPTREND"
    assert 0.0 <= o.confidence <= 1.0
    assert o.action is None
