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
        self.health_monitor = HealthMonitor()
        configure_logger(config.RUN_NAME)
        logging.info(
            f"🔧 Config: {config.CYCLE_MINUTES}min cycles, {config.LEVERAGE}x leverage")
        logging.info(
            f"💰 Position size: {config.POSITION_SIZE}, Stop loss: {config.STOP_LOSS_PERCENT}%")

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

    def _get_market_data_with_fallback(self, symbol: str):
        """Get market data with multiple fallback strategies"""
        try:
            return self._get_market_data(symbol)
        except Exception as e:
            logging.warning(
                f"⚠️ Primary market data failed, using fallback: {e}")
            # Fallback: use current price with neutral data
            price = self.exchange.get_current_price(symbol)
            return price, 0.0, 100_000_000  # Neutral change, average volume

    def _get_ai_analysis_with_timeout(self, price: float, change_pct: float, volume: float, symbol: str):
        """Get AI analysis with timeout protection"""
        def timeout_handler(signum, frame):
            raise TimeoutError("AI analysis timeout")
        # Set timeout to 30 seconds
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        try:
            prompt = build_prompt(price, change_pct, volume,
                                  self.config.CYCLE_MINUTES, symbol)
            outlook = send_request(prompt, crypto_symbol="Bitcoin")
            return outlook
        except TimeoutError:
            logging.error("⏰ AI analysis timed out after 30 seconds")
            return AIOutlook(interpretation="Neutral", reasons="Fallback: AI timeout")
        except Exception as e:
            logging.error(f"❌ AI analysis failed: {e}")
            return AIOutlook(interpretation="Neutral", reasons=f"Fallback: {str(e)}")
        finally:
            signal.alarm(0)  # Cancel timeout
            signal.signal(signal.SIGALRM, original_handler)

    def _execute_trading_decision(self, action: str, symbol: str, position):
        """Execute trading decision with detailed error handling"""
        try:
            # Configure exchange settings
            logging.info("⚙️ Configuring exchange settings...")
            self.exchange.set_margin_mode(symbol, self.config.MARGIN_MODE)
            self.exchange.set_leverage(symbol, self.config.LEVERAGE)
            # Execute trading action
            if action in ("OPEN_LONG", "REVERSE_TO_LONG"):
                if "REVERSE" in action and position:
                    logging.info(
                        "🔄 Reversing position: Closing current position")
                    self.exchange.flash_close_position(symbol)
                logging.info("📈 Opening LONG position")
                self.exchange.open_position(
                    symbol, "buy", self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT
                )
            elif action in ("OPEN_SHORT", "REVERSE_TO_SHORT"):
                if "REVERSE" in action and position:
                    logging.info(
                        "🔄 Reversing position: Closing current position")
                    self.exchange.flash_close_position(symbol)
                logging.info("📉 Opening SHORT position")
                self.exchange.open_position(
                    symbol, "sell", self.config.POSITION_SIZE, self.config.STOP_LOSS_PERCENT
                )
            elif action == "CLOSE" and position:
                logging.info("🔒 Closing position")
                self.exchange.flash_close_position(symbol)
            else:
                logging.info(
                    "💤 No action required - maintaining current position")
            self.health_monitor.record_cycle(True)
            logging.info("✅ Trade execution completed successfully")
        except Exception as e:
            self.health_monitor.record_cycle(False, str(e))
            # Enhanced error logging
            if hasattr(e, "last_attempt") and e.last_attempt is not None:
                root_cause = e.last_attempt.exception()
                logging.error(
                    f"💥 Trade execution FAILED → Root cause: {root_cause}")
            elif hasattr(e, "__cause__") and e.__cause__:
                logging.error(
                    f"💥 Trade execution FAILED → Cause: {e.__cause__}")
            else:
                logging.error(f"💥 Trade execution FAILED → {e}")
            # Log full traceback for debugging
            logging.debug("Full traceback:", exc_info=True)

    def run_cycle(self):
        logging.info("🔄 " + "=" * 50)
        logging.info("🔄 NEW TRADING CYCLE STARTED")
        logging.info("🔄 " + "=" * 50)
        symbol = self.config.SYMBOL
        # Get market data
        try:
            price, change_pct, volume = self._get_market_data_with_fallback(
                symbol)
            logging.info(
                f"📊 Market Data: ${price:,.2f} | Δ{change_pct:+.2f}% | Vol: ${volume:,.0f}")
        except Exception as e:
            logging.error(f"💥 Critical: Failed to fetch market data: {e}")
            self.health_monitor.record_cycle(False, f"Market data: {e}")
            return
        # Get AI analysis
        logging.info("🤖 Requesting AI analysis...")
        outlook = self._get_ai_analysis_with_timeout(
            price, change_pct, volume, symbol)
        interpretation = outlook.interpretation
        # Save AI response to logs directory
        save_response(outlook, self.config.RUN_NAME)
        # Display AI decision with emojis
        emoji = "📈" if interpretation == "Bullish" else "📉" if interpretation == "Bearish" else "➡️"
        logging.info(f"🎯 AI DECISION: {emoji} {interpretation:>8}")
        logging.info(f"💭 REASONING: {outlook.reasons}")
        # Get current position and make decision
        position = self.exchange.get_pending_positions(symbol)
        current_side = position.side if position else "FLAT"
        action = get_action(interpretation, current_side)
        # Display trading decision
        action_emojis = {
            "OPEN_LONG": "🟢📈",
            "OPEN_SHORT": "🔴📉",
            "REVERSE_TO_LONG": "🔄📈",
            "REVERSE_TO_SHORT": "🔄📉",
            "CLOSE": "🔒",
            "HOLD": "⏸️",
            "STAY_FLAT": "💤"
        }
        action_emoji = action_emojis.get(action, "❓")
        logging.info(f"⚡ TRADING ACTION: {action_emoji} {action:>15}")
        logging.info(f"📦 CURRENT POSITION: {current_side}")
        # Execute trading decision
        self._execute_trading_decision(action, symbol, position)
        # Log cycle completion
        logging.info("✅ " + "=" * 50)
        logging.info("✅ TRADING CYCLE COMPLETED")
        logging.info("✅ " + "=" * 50)
        logging.info("")  # Empty line for readability


class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.cycle_count = 0
        self.errors = []

    def record_cycle(self, success: bool, error: str = None):
        self.cycle_count += 1
        if not success and error:
            self.errors.append({
                'time': time.time(),
                'error': error,
                'cycle': self.cycle_count
            })
            logging.error(f"❌ Cycle {self.cycle_count} failed: {error}")
        else:
            logging.info(f"✅ Cycle {self.cycle_count} completed successfully")

    def get_status(self):
        uptime = time.time() - self.start_time
        error_rate = len(self.errors) / \
            self.cycle_count if self.cycle_count > 0 else 0
        status = {
            'uptime_hours': round(uptime / 3600, 2),
            'total_cycles': self.cycle_count,
            'error_count': len(self.errors),
            'error_rate': round(error_rate * 100, 2),
            'last_errors': self.errors[-5:] if self.errors else []
        }
        logging.info(f"📈 Health Status: {status['total_cycles']} cycles, "
                     f"{status['error_count']} errors ({status['error_rate']}% error rate)")
        return status
