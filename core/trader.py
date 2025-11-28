# core/trader.py
import logging
from config.settings import TradingConfig
from exchanges.base import BaseExchange
from ai.prompt_builder import build_prompt
from ai.provider import send_request, save_response, AIOutlook  # ← NEW IMPORT
from utils.helpers import open_position
from utils.logger import configure_logger
from core.decision_engine import get_action


class TradingBot:
    def __init__(self, config: TradingConfig, exchange: BaseExchange, llm_key: str = None):
        self.config = config
        self.exchange = exchange
        self.llm_key = llm_key or os.getenv("LLM_API_KEY")
        configure_logger(config.RUN_NAME)

    def run_cycle(self):
        logging.info("=== Trading Cycle Started ===")
        symbol = self.config.SYMBOL
        price = self.exchange.get_current_price(symbol)
        change_pct = 0.0   # You can improve this later with real candle data
        volume = "N/A"
        prompt = build_prompt(price, change_pct, volume,
                              self.config.CYCLE_MINUTES, symbol)
        try:
            outlook: AIOutlook = send_request(
                prompt, crypto_symbol="Bitcoin", api_key=self.llm_key)
            interpretation = outlook.interpretation
            save_response(outlook, self.config.RUN_NAME)
            logging.info(
                f"AI Outlook: {interpretation} — {outlook.reasons[:100]}...")
        except Exception as e:
            logging.error(f"AI failed: {e}")
            interpretation = "Neutral"
        position = self.exchange.get_pending_positions(symbol)
        current_side = position.side if position else None
        action = get_action(interpretation, current_side)
        # ... rest of execution logic unchanged ...
