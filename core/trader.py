# core/trader.py
import random
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional
from exchanges.base import BaseExchange
from ai.prompt_builder import build_prompt
from ai.provider import send_request, save_response, AIOutlook
from utils.logger import configure_logger
from utils.risk_manager import RiskManager
from tenacity import retry, stop_after_attempt, wait_fixed
import traceback


class TradingBot:
    def __init__(self, config: Dict[str, Any], exchange: BaseExchange):
        """Initialize the trading bot with configuration dictionary.

        Args:
            config: Dictionary containing all configuration parameters
            exchange: Exchange instance to use for trading
        """
        self.config = config
        self.exchange = exchange
        self.logger = logging.getLogger("trade")
        self.app_logger = logging.getLogger("app")
        self.risk_manager = RiskManager(self.config, self.exchange)
        self.current_position = None
        self.last_trade_time = None
        self.start_time = datetime.now()

    async def _update_position_from_exchange(self):
        """Update local position state from exchange using get_pending_positions."""
        try:
            position = await self.exchange.get_pending_positions(self.config['SYMBOL'])

            if position:
                # Convert timestamp to datetime, handling both seconds and milliseconds
                if position.timestamp:
                    # If timestamp is in milliseconds (typical for exchanges), convert by dividing by 1000
                    # If it's already in seconds, then dividing by 1000 would be wrong. We'll assume milliseconds if > 1e10
                    if position.timestamp > 1e10:  # Likely milliseconds
                        timestamp = datetime.fromtimestamp(position.timestamp / 1000)
                    else:
                        timestamp = datetime.fromtimestamp(position.timestamp)
                else:
                    timestamp = datetime.now()

                self.current_position = {
                    'side': position.side,
                    'quantity': position.size,
                    'entry_price': position.entry_price,
                    'timestamp': timestamp
                }
                self.last_trade_time = timestamp
                self.app_logger.info(f"Detected existing position: {position.side} {position.size} @ ${position.entry_price:,.0f}")
            else:
                if self.current_position is not None:
                    self.app_logger.info("Position closed - no positions found")
                self.current_position = None
                self.last_trade_time = None
        except Exception as e:
            self.logger.warning(f"Failed to update position from exchange: {e}")
            # Keep existing position state on error rather than clearing it

    async def run_cycle(self):
        """Run a single trading cycle: analyze market, get AI decision, execute trade if appropriate."""
        try:
            # Update local position state from exchange
            await self._update_position_from_exchange()
            # Monitor current position for SL/TP/hold time
            await self.monitor_positions()

            # Log cycle start with key info
            account_balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
            position_status = f"{self.current_position['side']} {self.current_position['quantity']}@{self.current_position['entry_price']:.0f}" if self.current_position else "No position"
            self.app_logger.info(f"Cycle start | Equity: ${account_balance:,.0f} | Position: {position_status}")

            # Fetch market data
            ohlcv = await self.exchange.get_ohlcv(
                self.config['SYMBOL'],
                timeframe=self.config['CYCLE_MINUTES'],
                limit=100
            )
            if not ohlcv or len(ohlcv) < 50:
                self.logger.warning("Insufficient market data")
                return

            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(
                df['timestamp'].astype(int), unit='ms')
            df.set_index('timestamp', inplace=True)

            # Calculate technical indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            std20 = df['close'].rolling(window=20).std()
            df['bb_middle'] = df['sma_20']
            df['bb_upper'] = df['sma_20'] + (2 * std20)
            df['bb_lower'] = df['sma_20'] - (2 * std20)
            df['ema_20'] = ta.ema(df['close'], length=20)
            macd = ta.macd(df['close'])
            df['MACD_hist'] = macd['MACDh_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']

            current_price = df['close'].iloc[-1]

            # Fetch live price for comparison
            live_price = await self.exchange.get_current_price(self.config['SYMBOL'])
            price_diff = live_price - current_price
            price_diff_pct = (price_diff / current_price) * 100 if current_price != 0 else 0

            latest_indicators = {
                'rsi': df['rsi'].iloc[-1],
                'sma_20': df['sma_20'].iloc[-1],
                'sma_50': df['sma_50'].iloc[-1],
                'bb_position': (current_price - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) if df['bb_upper'].iloc[-1] != df['bb_lower'].iloc[-1] else 0.5,
                'price': current_price,
                'live_price': live_price,
                'price_diff': price_diff,
                'price_diff_pct': price_diff_pct,
                'trend': 'bullish' if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else 'bearish',
                'RSI': df['rsi'].iloc[-1],
                'EMA_20': df['ema_20'].iloc[-1],
                'BB_upper': df['bb_upper'].iloc[-1],
                'BB_middle': df['bb_middle'].iloc[-1],
                'BB_lower': df['bb_lower'].iloc[-1],
                'MACD': {
                    'hist': df['MACD_hist'].iloc[-1],
                    'signal': df['MACD_signal'].iloc[-1]
                }
            }
            # Get AI outlook
            timestamp = datetime.now()
            minutes_elapsed = int((datetime.now() - self.start_time).total_seconds() // 60)
            account_balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
            equity = account_balance  # TODO: include unrealized PnL if any
            open_positions = [
                self.current_position] if self.current_position else []
            price_history_short = df.tail(20).reset_index().to_dict('records')
            price_history_long = df.tail(50).reset_index().to_dict('records')
            indicators = latest_indicators
            predictive_signals = {}
            prompt = build_prompt(
                timestamp, minutes_elapsed, account_balance, equity, open_positions,
                price_history_short, price_history_long, indicators, predictive_signals, self.config
            )
            outlook = await send_request(prompt, self.config)
            save_response(outlook, self.config['RUN_NAME'])

            # Parse AI outlook
            ai_decision = self._parse_ai_outlook(outlook)

            # Log AI decision with futures terminology
            ai_action_display = {
                'BUY': 'OPEN_LONG',
                'SELL': 'OPEN_SHORT',
                'CLOSE_POSITION': 'CLOSE_POSITION',
                'HOLD': 'HOLD',
                'NO_TRADE': 'NO_TRADE'
            }.get(ai_decision['action'], ai_decision['action'])

            # Log AI decision
            self.app_logger.info(f"AI Decision: {ai_action_display} | Confidence: {ai_decision['confidence']:.2f} | Price: ${current_price:,.1f}")

            # Check risk management
            if not await self.risk_manager.can_trade(ai_decision, current_price, self.current_position):
                self.logger.info(
                    f"Action: {ai_action_display}, confidence: {ai_decision['confidence']:.2f}")
                return

            # Execute trade or close position
            if ai_decision['action'] in ['BUY', 'SELL']:
                await self._execute_trade(ai_decision, current_price)
            elif ai_decision['action'] == 'CLOSE_POSITION':
                if self.current_position:
                    await self._close_position(self.current_position['side'], current_price, "AI decision")
                else:
                    self.logger.warning("AI requested CLOSE_POSITION but no position is open")

            self.logger.info(
                f"Cycle completed. AI Decision: {ai_action_display}")

        except Exception as e:
            self.logger.error(f"Error in run_cycle: {e}")
            raise

    def _parse_ai_outlook(self, outlook: AIOutlook) -> Dict[str, Any]:
        """Parse AI outlook into actionable decision."""
        # Use the action provided by the AI, or map interpretation to action if action is not provided
        if outlook.action and outlook.action in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_POSITION']:
            # Map futures actions to exchange actions
            action_map = {
                'OPEN_LONG': 'BUY',
                'OPEN_SHORT': 'SELL',
                'CLOSE_POSITION': 'CLOSE_POSITION'
            }
            action = action_map[outlook.action]
        else:
            # Map interpretation to action - new assertive logic
            if outlook.interpretation == 'STRONG_UPTREND':
                action = 'BUY'
            elif outlook.interpretation == 'STRONG_DOWNTREND':
                action = 'SELL'
            else:
                action = 'BUY'  # Default to buying in uptrend

        # Use provided confidence, or set defaults based on action
        if outlook.confidence is not None and 0 <= outlook.confidence <= 1:
            confidence = outlook.confidence
        else:
            confidence = 0.8 if action in ['BUY', 'SELL'] else 0.5

        reason = outlook.reasons
        return {'action': action, 'confidence': confidence, 'reason': reason}

    async def _execute_trade(self, decision: Dict[str, Any], current_price: float):
        """Execute a trade based on AI decision."""
        try:
            symbol = self.config['SYMBOL']
            side = decision['action']
            quantity = await self._calculate_position_size(current_price)

            if quantity <= 0:
                self.logger.warning("Calculated position size is zero or negative")
                return

            # Check for existing positions directly from exchange before placing order
            existing_position = await self.exchange.get_pending_positions(symbol)
            if existing_position:
                self.logger.warning(f"Existing position found on exchange: {existing_position.side} {existing_position.size} @ ${existing_position.entry_price:,.0f} - skipping new order")
                # Update local state to match exchange
                self.current_position = {
                    'side': existing_position.side,
                    'quantity': existing_position.size,
                    'entry_price': existing_position.entry_price,
                    'timestamp': datetime.fromtimestamp(existing_position.timestamp / 1000) if existing_position.timestamp > 1e10 else datetime.fromtimestamp(existing_position.timestamp)
                }
                return

            # Execute the trade
            order = await self.exchange.open_position(
                symbol=symbol,
                side=side,
                size=str(quantity)
            )

            # Debug: Log order response details
            self.logger.info(f"Order response: {order}")

            if order:
                # Verify position was actually opened by fetching from exchange
                await asyncio.sleep(1)  # Brief delay to allow order processing
                verified_position = await self.exchange.get_pending_positions(symbol)
                if verified_position and verified_position.side == side and abs(verified_position.size - quantity) < 0.001:
                    # Position verified - update local state
                    self.current_position = {
                        'side': side,
                        'quantity': quantity,
                        'entry_price': verified_position.entry_price,
                        'timestamp': datetime.fromtimestamp(verified_position.timestamp / 1000) if verified_position.timestamp > 1e10 else datetime.fromtimestamp(verified_position.timestamp)
                    }
                    self.last_trade_time = datetime.now()

                    self.logger.info(f"Trade executed and verified: {side} {quantity} {symbol} at ${verified_position.entry_price:,.1f}")
                    balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
                    trade = {
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'side': side,
                        'quantity': quantity,
                        'price': verified_position.entry_price,
                        'pnl': 0.0,
                        'balance': balance
                    }
                    self.risk_manager.update_trade_history(trade)
                else:
                    self.logger.error(f"Position verification failed. Expected: {side} {quantity}, Got: {verified_position.side if verified_position else 'None'} {verified_position.size if verified_position else 'N/A'}")
                    # Reset local state on verification failure
                    self.current_position = None
            else:
                self.logger.error(f"Order failed: {order}")
                self.current_position = None

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            # Reset local position state on error
            self.current_position = None

    async def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on config and risk."""
        balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
        leverage = self.config.get('LEVERAGE', 1)

        # Apply risk management percentage to balance first
        max_risk_pct = self.config.get('MAX_RISK_PERCENT', 1.0)
        max_position_value = balance * (max_risk_pct / 100.0)

        # Calculate position value considering leverage
        # For futures: position_value = max_position_value * leverage
        position_value = max_position_value * leverage
        quantity = position_value / price

        # Round to 4 decimal places for precision
        quantity = round(quantity, 4)

        self.app_logger.info(f"Position size: {quantity} {self.config['SYMBOL']} | Balance: ${balance:,.0f} | Max Risk: {max_risk_pct:.1f}% | Leverage: {leverage}x | Price: ${price:,.1f}")
        return quantity

    async def monitor_positions(self):
        """Monitor open positions for SL/TP triggers and max hold time."""
        if not self.current_position:
            return

        try:
            current_price = await self.exchange.get_current_price(self.config['SYMBOL'])
            entry_price = self.current_position['entry_price']
            side = self.current_position['side']

            # Calculate current PnL
            if side == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:  # SELL
                pnl_pct = (entry_price - current_price) / entry_price * 100

            # Check stop-loss
            sl_threshold = -self.config.get('STOP_LOSS_PERCENT', 2.0)
            if pnl_pct <= sl_threshold:
                self.logger.warning(f"Stop-loss triggered: {pnl_pct:.2f}% loss (threshold: {sl_threshold:.1f}%)")
                await self._close_position(side, current_price, f"Stop-loss at {sl_threshold:.1f}%")
                return

            # Check take-profit
            tp_threshold = self.config.get('TAKE_PROFIT_PERCENT', 4.0)
            if pnl_pct >= tp_threshold:
                self.logger.info(f"Take-profit triggered: {pnl_pct:.2f}% profit (threshold: {tp_threshold:.1f}%)")
                await self._close_position(side, current_price, f"Take-profit at {tp_threshold:.1f}%")
                return

            # Check max hold time
            position_age_hours = (datetime.now() - self.current_position['timestamp']).total_seconds() / 3600
            max_hold_hours = self.config.get('MAX_POSITION_HOLD_HOURS', 24)
            if position_age_hours >= max_hold_hours:
                self.logger.info(f"Max hold time reached: {position_age_hours:.1f} hours")
                await self._close_position(side, current_price, f"Max hold time ({max_hold_hours}h)")
                return

            # Log position status periodically
            self.app_logger.debug(f"Position monitoring: {side} | Entry: ${entry_price:,.1f} | Current: ${current_price:,.1f} | PnL: {pnl_pct:+.2f}%")

        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    async def _close_position(self, side: str, price: float, reason: str):
        """Close current position with validation."""
        if not self.current_position:
            return

        try:
            symbol = self.config['SYMBOL']
            expected_quantity = self.current_position['quantity']
            expected_side = self.current_position['side']

            # Get the current position from exchange and validate it matches expected
            position = await self.exchange.get_pending_positions(symbol)
            if not position:
                self.logger.warning(f"No open position found for {symbol}")
                self.current_position = None
                return

            # Validate position matches expected
            if abs(position.size - expected_quantity) > 0.001 or position.side != expected_side:
                self.logger.warning(f"Position mismatch - Expected: {expected_side} {expected_quantity}, Got: {position.side} {position.size}. Closing current position anyway.")
                # Update expected values to match current position for closing
                expected_quantity = position.size
                expected_side = position.side

            # Close the validated position
            result = await self.exchange.close_position(position, reason)
            pnl = result.get('pnl', 0.0)

            if result.get('order'):
                self.logger.info(f"Position closed: {reason} at ${price:,.1f} | PnL: ${pnl:+.2f}")
                self.current_position = None
                self.risk_manager.update_daily_pnl(pnl)
                # Fetch current balance for trade record
                balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
                trade = {
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'side': f'CLOSE_{expected_side}',
                    'quantity': expected_quantity,
                    'price': price,
                    'pnl': pnl,
                    'balance': balance
                }
                self.risk_manager.update_trade_history(trade)
            else:
                self.logger.error(f"Failed to close position")

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            # Don't reset position state on error to avoid state corruption
