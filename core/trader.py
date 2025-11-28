# core/trader.py
import logging
from config.settings import TradingConfig
from exchanges.base import BaseExchange
from ai.prompt_builder import build_prompt
from ai.provider import send_request, save_response, AIOutlook
from utils.logger import configure_logger
from core.decision_engine import get_action
from tenacity import retry, stop_after_attempt, wait_fixed


class TradingBot:
    def __init__(self, config: TradingConfig, exchange: BaseExchange):
        self.config = config
        self.exchange = exchange
        configure_logger(config.RUN_NAME)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(3))
    def _get_market_data(self, symbol: str):
        # Paper trading: generate fake but realistic data
        if self.config.FORWARD_TESTING:
            price = self.exchange.get_current_price(symbol)
            change_pct = random.uniform(-2.5, 2.5)
            volume = random.uniform(50_000_000, 500_000_000)
            return price, change_pct, volume
        # Live: real data
        ohlcv = self.exchange.exchange.fetch_ohlcv(
            symbol, timeframe='1m', limit=15)
        current_price = ohlcv[-1][4]
        prev_price = ohlcv[-11][4] if len(ohlcv) >= 11 else current_price
        change_pct = (current_price - prev_price) / prev_price * 100
        volume = ohlcv[-1][5]
        return current_price, change_pct, volume

    def run_cycle(self):
        logging.info("=== New Trading Cycle ===")
        symbol = self.config.SYMBOL
        try:
            price, change_pct, volume = self._get_market_data(symbol)
        except Exception as e:
            logging.error(f"Failed to fetch market data: {e}")
            return
        prompt = build_prompt(
            price=price,
            change_pct=change_pct,
            volume=volume,
            cycle=self.config.CYCLE_MINUTES,
            symbol=symbol
        )
        outlook: AIOutlook = send_request(prompt, crypto_symbol="Bitcoin")
        interpretation = outlook.interpretation
        save_response(outlook, self.config.RUN_NAME)
        logging.info(
            f"AI → {interpretation:>8} | ${price:,.1f} | Δ{change_pct:+.2f}%")
        position = self.exchange.get_pending_positions(symbol)
        current_side = position.side if position else None
        action = get_action(interpretation, current_side)
        logging.info(
            f"Decision → {action:>15} | Position: {current_side or 'FLAT'}")
        try:
            self.exchange.set_margin_mode(symbol, self.config.MARGIN_MODE)
            self.exchange.set_leverage(symbol, self.config.LEVERAGE)
            if action in ("OPEN_LONG", "REVERSE_TO_LONG"):
                if "REVERSE" in action and position:
                    self.exchange.flash_close_position(symbol)
                self.exchange.open_position(
                    symbol, "buy", self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT)
            elif action in ("OPEN_SHORT", "REVERSE_TO_SHORT"):
                if "REVERSE" in action and position:
                    self.exchange.flash_close_position(symbol)
                self.exchange.open_position(
                    symbol, "sell", self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT)
            elif action == "CLOSE" and position:
                self.exchange.flash_close_position(symbol)
        except Exception as e:
            logging.error(f"Trade execution failed: {e}")
        logging.info("Cycle completed\n")
