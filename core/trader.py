# core/trader.py
import logging
import os
from config.settings import TradingConfig
from exchanges.base import BaseExchange
from ai.prompt_builder import build_prompt
from ai.provider import send_request, save_response, AIOutlook
from utils.logger import configure_logger
from utils.helpers import open_position
from core.decision_engine import get_action


class TradingBot:
    def __init__(self, config: TradingConfig, exchange: BaseExchange):
        self.config = config
        self.exchange = exchange
        self.llm_key = os.getenv("LLM_API_KEY")
        configure_logger(config.RUN_NAME)

    def run_cycle(self):
        logging.info("=== New Trading Cycle ===")
        symbol = self.config.SYMBOL
        # Get price
        try:
            price = self.exchange.get_current_price(symbol)
        except Exception as e:
            logging.error(f"Failed to get price: {e}")
            return
        # Build prompt & send prompt
        prompt = build_prompt(price, change_pct=0.0, volume="—",
                              cycle=self.config.CYCLE_MINUTES, symbol=symbol)
        try:
            outlook: AIOutlook = send_request(
                prompt, crypto_symbol="Bitcoin", api_key=self.llm_key)
            interpretation = outlook.interpretation
            save_response(outlook, self.config.RUN_NAME)
            logging.info(f"AI → {interpretation:>8} | ${price:,.1f}")
        except Exception as e:
            logging.error(f"AI failed ({e}) → Neutral")
            interpretation = "Neutral"
        # Current position
        position = self.exchange.get_pending_positions(symbol)
        current_side = position.side if position else None
        # Decide & execute
        action = get_action(interpretation, current_side)
        logging.info(
            f"Decision → {action:>15} | Position: {current_side or 'flat'}")
        try:
            self.exchange.set_margin_mode(symbol, self.config.MARGIN_MODE)
            self.exchange.set_leverage(symbol, self.config.LEVERAGE)
            if action in ("OPEN_LONG", "REVERSE_TO_LONG"):
                if "REVERSE" in action and position:
                    self.exchange.flash_close_position(position.positionId)
                open_position(self.exchange, symbol, "buy",
                              self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT)
            elif action in ("OPEN_SHORT", "REVERSE_TO_SHORT"):
                if "REVERSE" in action and position:
                    self.exchange.flash_close_position(position.positionId)
                open_position(self.exchange, symbol, "sell",
                              self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT)
            elif action == "CLOSE" and position:
                self.exchange.flash_close_position(position.positionId)
        except Exception as e:
            logging.error(f"Trade execution failed: {e}")
        logging.info("Cycle completed\n")
