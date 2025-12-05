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
        """Get market data including cycle-specific volume"""
        cycle_minutes = self.config.CYCLE_MINUTES
        if self.config.FORWARD_TESTING:
            price = self.exchange.get_current_price(symbol)
            change_pct = random.uniform(-2.5, 2.5)
            # For paper trading, generate realistic volumes
            volume_24h = random.uniform(200_000_000, 500_000_000)
            # Cycle volume should be roughly (cycle_minutes/1440) of 24h volume
            volume_cycle = volume_24h * \
                (cycle_minutes / 1440) * random.uniform(0.5, 1.5)
            return price, change_pct, volume_24h, volume_cycle
        # Calculate candles needed for the cycle
        timeframe = self._determine_timeframe(cycle_minutes)
        candles_needed = self._calculate_candles_needed(
            cycle_minutes, timeframe)
        # Fetch OHLCV data
        ohlcv = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=candles_needed)
        if len(ohlcv) < 2:
            raise ValueError(
                f"Not enough data for {cycle_minutes}-minute analysis")
        # Current price from last candle
        current_price = ohlcv[-1][4]
        # Calculate price change over the cycle
        price_change = self._calculate_price_change(
            ohlcv, cycle_minutes, timeframe)
        # Calculate 24h volume (approximate from multiple candles)
        volume_24h = self._calculate_24h_volume(ohlcv, timeframe)
        # Calculate volume for the cycle period
        volume_cycle = self._calculate_cycle_volume(
            ohlcv, cycle_minutes, timeframe)
        return current_price, price_change, volume_24h, volume_cycle

    def _determine_timeframe(self, cycle_minutes: int) -> str:
        """Determine optimal timeframe based on cycle length"""
        if cycle_minutes <= 15:
            return '1m'
        elif cycle_minutes <= 60:
            return '5m'
        elif cycle_minutes <= 240:  # 4 hours
            return '15m'
        else:
            return '1h'

    def _calculate_candles_needed(self, cycle_minutes: int, timeframe: str) -> int:
        """Calculate how many candles to fetch based on timeframe"""
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }.get(timeframe, 60)
        # Fetch enough for cycle + extra for calculations
        candles_for_cycle = (cycle_minutes // timeframe_minutes) + 1
        # Also fetch enough for 24h volume calculation
        candles_for_24h = (1440 // timeframe_minutes) + 1
        # Ensure minimum 100 candles
        return max(candles_for_cycle, candles_for_24h, 100)

    def _calculate_price_change(self, ohlcv: list, cycle_minutes: int, timeframe: str) -> float:
        """Calculate price change over the specified cycle period"""
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }.get(timeframe, 60)
        # Calculate how many candles back we need for the cycle
        candles_back = cycle_minutes // timeframe_minutes
        if candles_back >= len(ohlcv):
            candles_back = len(ohlcv) - 1
        current_price = ohlcv[-1][4]
        past_price = ohlcv[-candles_back -
                           1][4] if candles_back < len(ohlcv) else ohlcv[0][4]
        return ((current_price - past_price) / past_price) * 100

    def _calculate_24h_volume(self, ohlcv: list, timeframe: str) -> float:
        """Calculate approximate 24h volume"""
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }.get(timeframe, 60)
        # Calculate how many candles in 24 hours
        candles_24h = 1440 // timeframe_minutes
        # Sum volume from available candles (up to 24h worth)
        volume_sum = 0
        for i in range(min(candles_24h, len(ohlcv))):
            volume_sum += ohlcv[-(i+1)][5]
        return volume_sum

    def _calculate_cycle_volume(self, ohlcv: list, cycle_minutes: int, timeframe: str) -> float:
        """Calculate volume for the specific cycle period"""
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }.get(timeframe, 60)
        # Calculate how many candles in the cycle
        candles_in_cycle = cycle_minutes // timeframe_minutes
        # Sum volume from the cycle period
        volume_sum = 0
        for i in range(min(candles_in_cycle, len(ohlcv))):
            volume_sum += ohlcv[-(i+1)][5]
        return volume_sum

    def _get_ai_analysis_with_timeout(self, price: float, change_pct: float,
                                      volume_24h: float, volume_cycle: float, symbol: str):
        """Get AI analysis with timeout protection"""
        def timeout_handler(signum, frame):
            raise TimeoutError("AI analysis timeout")
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        try:
            prompt = build_prompt(price, change_pct, volume_24h, volume_cycle,
                                  self.config.CYCLE_MINUTES, symbol)
            # Log the complete AI prompt being sent
            logging.info("🤖 AI PROMPT SENT:")
            # Split prompt into lines and log each one
            prompt_lines = prompt.strip().split('\n')
            for line in prompt_lines:
                logging.info(f"   {line}")
            logging.info("")  # Empty line for separation
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
                    logging.info(f"🔄 Reversing from SHORT to LONG")
                    self.exchange.flash_close_position(symbol)
                position_value = self._calculate_position_value(
                    current_balance)
                self.exchange.open_position(
                    symbol, "buy", self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT
                )
            elif action in ("OPEN_SHORT", "REVERSE_TO_SHORT"):
                if "REVERSE" in action and position:
                    logging.info(f"🔄 Reversing from LONG to SHORT")
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
        # Get market data (now includes both 24h and cycle volume)
        try:
            price, change_pct, volume_24h, volume_cycle = self._get_market_data(
                symbol)
        except Exception as e:
            logging.error(f"Market data error: {e}")
            return
        # Get AI analysis
        outlook = self._get_ai_analysis_with_timeout(
            price, change_pct, volume_24h, volume_cycle, symbol)
        interpretation = outlook.interpretation
        save_response(outlook, self.config.RUN_NAME)
        # Get current position
        position = self.exchange.get_pending_positions(symbol)
        current_side = position.side if position else "FLAT"
        action = get_action(interpretation, current_side)
        # Log decision with context
        cycle_label = f"{self.config.CYCLE_MINUTES}min"
        if position:
            current_price = self.exchange.get_current_price(symbol)
            pnl = (current_price - position.entry_price) * position.size
            pnl_pct = ((current_price - position.entry_price) /
                       position.entry_price) * 100
            logging.info(
                f"${price:,.0f} | Δ{change_pct:+.1f}% ({cycle_label}) | AI: {interpretation} | Pos: {current_side} (${pnl:+.0f}) → {action}")
        else:
            logging.info(
                f"${price:,.0f} | Δ{change_pct:+.1f}% ({cycle_label}) | AI: {interpretation} | Pos: {current_side} → {action}")
        # Execute trade
        self._execute_trading_decision(action, symbol, position)
