# core/trader.py
import random
import logging
import time
import signal
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
        self.current_balance = config.INITIAL_CAPITAL
        configure_logger(config.RUN_NAME)
        mode = "PAPER" if config.FORWARD_TESTING else "LIVE"
        logging.info(
            f"{mode} | {config.SYMBOL} | {config.CYCLE_MINUTES}min | {config.LEVERAGE}x")
        logging.info(f"AI: {config.LLM_PROVIDER} ({config.LLM_MODEL})")

    def _get_effective_balance(self) -> float:
        """Get the current balance (live or paper)"""
        if self.config.FORWARD_TESTING:
            return self.config.INITIAL_CAPITAL
        try:
            fresh_balance = self.exchange.get_account_balance(
                self.config.CURRENCY)
            self.current_balance = fresh_balance
            return fresh_balance
        except Exception:
            return self.current_balance

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(3))
    def _get_market_data(self, symbol: str):
        if self.config.FORWARD_TESTING:
            price = self.exchange.get_current_price(symbol)
            change_pct = random.uniform(-2.5, 2.5)
            volume = random.uniform(50_000_000, 500_000_000)
            return price, change_pct, volume
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=15)
        current_price = ohlcv[-1][4]
        prev_price = ohlcv[-11][4] if len(ohlcv) >= 11 else current_price
        change_pct = (current_price - prev_price) / prev_price * 100
        volume = ohlcv[-1][5]
        return current_price, change_pct, volume

    def _get_ai_analysis_with_timeout(self, price: float, change_pct: float, volume: float, symbol: str):
        """Get AI analysis with timeout protection"""
        def timeout_handler(signum, frame):
            raise TimeoutError("AI analysis timeout")
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        try:
            prompt = build_prompt(price, change_pct, volume,
                                  self.config.CYCLE_MINUTES, symbol)
            outlook = send_request(prompt, self.config)
            return outlook
        except TimeoutError:
            logging.warning("AI timeout - using neutral")
            return AIOutlook(interpretation="Neutral", reasons="AI timeout")
        except Exception as e:
            logging.warning(f"AI error: {e}")
            return AIOutlook(interpretation="Neutral", reasons=f"AI error: {e}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

    def _calculate_position_value(self, balance: float) -> float:
        """Calculate position value based on config"""
        if self.config.POSITION_SIZE.endswith('%'):
            percentage = float(self.config.POSITION_SIZE[:-1]) / 100
            return balance * percentage
        else:
            return float(self.config.POSITION_SIZE)

    def _execute_trading_decision(self, action: str, symbol: str, position):
        """Execute trading decision with position awareness"""
        try:
            current_balance = self._get_effective_balance()
            # Get current position details for better decision making
            current_position = self.exchange.get_pending_positions(symbol)
            # Set exchange parameters
            self.exchange.set_margin_mode(symbol, self.config.MARGIN_MODE)
            self.exchange.set_leverage(symbol, self.config.LEVERAGE)
            if action in ("OPEN_LONG", "REVERSE_TO_LONG"):
                if "REVERSE" in action and position:
                    logging.info("🔄 Reversing from SHORT to LONG")
                    self.exchange.flash_close_position(symbol)
                position_value = self._calculate_position_value(
                    current_balance)
                self.exchange.open_position(
                    symbol, "buy", self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT
                )
            elif action in ("OPEN_SHORT", "REVERSE_TO_SHORT"):
                if "REVERSE" in action and position:
                    logging.info("🔄 Reversing from LONG to SHORT")
                    self.exchange.flash_close_position(symbol)
                position_value = self._calculate_position_value(
                    current_balance)
                self.exchange.open_position(
                    symbol, "sell", self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT
                )
            elif action == "CLOSE" and position:
                logging.info("🔒 Closing position")
                self.exchange.flash_close_position(symbol)
            else:
                if position:
                    # Calculate PnL for existing position
                    current_price = self.exchange.get_current_price(symbol)
                    pnl = (current_price - position.entry_price) * \
                        position.size
                    pnl_pct = ((current_price - position.entry_price) /
                               position.entry_price) * 100
                    logging.info(
                        f"⏸️ Holding {position.side} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
                else:
                    logging.info("💤 Staying flat")
        except Exception as e:
            logging.error(f"Trade failed: {e}")

    def run_cycle(self):
        symbol = self.config.SYMBOL
        # Get market data
        try:
            price, change_pct, volume = self._get_market_data(symbol)
        except Exception as e:
            logging.error(f"Market data error: {e}")
            return
        # Get AI analysis
        outlook = self._get_ai_analysis_with_timeout(
            price, change_pct, volume, symbol)
        interpretation = outlook.interpretation
        save_response(outlook, self.config.RUN_NAME)
        # Get current position
        position = self.exchange.get_pending_positions(symbol)
        current_side = position.side if position else "FLAT"
        action = get_action(interpretation, current_side)
        # Log decision with context
        if position:
            current_price = self.exchange.get_current_price(symbol)
            pnl = (current_price - position.entry_price) * position.size
            pnl_pct = ((current_price - position.entry_price) /
                       position.entry_price) * 100
            logging.info(
                f"${price:,.0f} | Δ{change_pct:+.1f}% | AI: {interpretation} | Pos: {current_side} (${pnl:+.0f}) → {action}")
        else:
            logging.info(
                f"${price:,.0f} | Δ{change_pct:+.1f}% | AI: {interpretation} | Pos: {current_side} → {action}")
        # Execute trade
        self._execute_trading_decision(action, symbol, position)
