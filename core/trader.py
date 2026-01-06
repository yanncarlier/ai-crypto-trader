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
        self._operation_lock = asyncio.Lock()  # Mutex to prevent concurrent operations
        self._state_monitor_task = None  # Background state monitoring task

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

                # Use consistent position object structure
                self.current_position = {
                    'side': position.side,
                    'quantity': position.size,
                    'entry_price': position.entry_price,
                    'timestamp': timestamp,
                    'sl_order_id': getattr(position, 'sl_order_id', None),
                    'tp_order_id': getattr(position, 'tp_order_id', None),
                    'position_id': position.positionId
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
            # Simple state synchronization on cycle start
            await self._update_position_from_exchange()

            # Monitor current position for SL/TP
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

            # Execute trade or close position with mutex protection
            async with self._operation_lock:
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
            # Map interpretation to action - conservative logic
            if outlook.interpretation == 'STRONG_UPTREND':
                action = 'BUY'
            elif outlook.interpretation == 'STRONG_DOWNTREND':
                action = 'SELL'
            else:
                action = 'HOLD'  # Default to hold for neutral/unknown signals

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
            # Update position state before trade execution
            await self._update_position_from_exchange()

            symbol = self.config['SYMBOL']
            side = decision['action']
            quantity = await self._calculate_position_size(current_price)

            if quantity <= 0:
                self.logger.warning("Calculated position size is zero or negative")
                return

            # Check for existing positions and handle reversals
            existing_position = await self.exchange.get_pending_positions(symbol)
            if existing_position:
                # Handle position reversals - if AI wants opposite direction, close existing first
                if ((side == 'BUY' and existing_position.side == 'SELL') or
                    (side == 'SELL' and existing_position.side == 'BUY')):
                    self.logger.info(f"Position reversal detected: closing {existing_position.side} to open {side}")
                    # Create a position dict for _close_position
                    position_dict = {
                        'side': existing_position.side,
                        'quantity': existing_position.size,
                        'entry_price': existing_position.entry_price,
                        'sl_order_id': getattr(existing_position, 'sl_order_id', None),
                        'tp_order_id': getattr(existing_position, 'tp_order_id', None)
                    }
                    await self._close_position(position_dict['side'], await self.exchange.get_current_price(symbol), "Position reversal")
                    # After closing, continue to place the new order - don't return
                    self.logger.info(f"Position closed. Now placing new {side} order.")
                else:
                    # Same direction - this shouldn't happen with proper AI logic, but handle gracefully
                    self.logger.warning(f"AI requested {side} but already in {existing_position.side} position - blocking duplicate position")
                    # Update local state to match exchange
                    self.current_position = {
                        'side': existing_position.side,
                        'quantity': existing_position.size,
                        'entry_price': existing_position.entry_price,
                        'timestamp': datetime.fromtimestamp(existing_position.timestamp / 1000) if existing_position.timestamp > 1e10 else datetime.fromtimestamp(existing_position.timestamp),
                        'sl_order_id': getattr(existing_position, 'sl_order_id', None),
                        'tp_order_id': getattr(existing_position, 'tp_order_id', None)
                    }
                    return

            # Execute the trade with retry mechanism
            max_retries = 3
            retry_delay = 1  # Start with 1 second delay

            for attempt in range(max_retries):
                try:
                    order = await self.exchange.open_position(
                        symbol=symbol,
                        side=side,
                        size=str(quantity)
                    )

                    # Debug: Log order response details
                    self.logger.info(f"Order response (attempt {attempt + 1}): {order}")

                    if order:
                        break  # Success, exit retry loop

                except Exception as order_error:
                    self.logger.warning(f"Order attempt {attempt + 1} failed: {order_error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        self.logger.error(f"All {max_retries} order attempts failed")
                        order = None

            if order:
                # Verify position was actually opened by checking order status and position data
                verified_position = None
                order_filled = False
                size_tolerance = quantity * 0.05  # 5% tolerance for size differences

                # First, verify the order was filled by checking order status
                order_id = order.get('orderId')
                if order_id:
                    for attempt in range(10):  # Try up to 10 times with increasing delays
                        await asyncio.sleep(min(1 + attempt * 0.5, 5))  # 1s, 1.5s, 2s, ... up to 5s

                        try:
                            # Check if order exists and is filled
                            order_status = await self.exchange.get_order_status(symbol, order_id)
                            if order_status and order_status.get('status') == 'filled':
                                order_filled = True
                                self.logger.info(f"Order {order_id} confirmed filled on attempt {attempt + 1}")
                                break
                            elif order_status and order_status.get('status') in ['canceled', 'rejected']:
                                self.logger.error(f"Order {order_id} was {order_status.get('status')} - aborting verification")
                                break
                        except Exception as e:
                            self.logger.warning(f"Failed to check order status on attempt {attempt + 1}: {e}")

                    if not order_filled:
                        self.logger.warning(f"Order {order_id} status could not be confirmed as filled")

                # Then verify position exists (with or without order confirmation)
                for attempt in range(8):  # Try up to 8 times with increasing delays
                    await asyncio.sleep(min(2 + attempt * 0.5, 8))  # 2s, 2.5s, 3s, ... up to 8s

                    verified_position = await self.exchange.get_pending_positions(symbol)
                    if (verified_position and verified_position.side == side and
                        abs(verified_position.size - quantity) <= size_tolerance):
                        break  # Verification successful
                    verified_position = None  # Reset for retry

                if verified_position:
                    # Position verified - update local state
                    self.current_position = {
                        'side': side,
                        'quantity': quantity,
                        'entry_price': verified_position.entry_price,
                        'timestamp': datetime.fromtimestamp(verified_position.timestamp / 1000) if verified_position.timestamp > 1e10 else datetime.fromtimestamp(verified_position.timestamp)
                    }
                    self.last_trade_time = datetime.now()

                    # Create conditional SL/TP orders
                    sl_pct = self.config.get('STOP_LOSS_PERCENT', 2.0)
                    tp_pct = self.config.get('TAKE_PROFIT_PERCENT', 4.0)
                    conditional_results = await self.exchange._create_conditional_orders(
                        symbol, quantity, side, verified_position.entry_price, sl_pct, tp_pct
                    )

                    # Store SL/TP order IDs in position state
                    self.current_position['sl_order_id'] = conditional_results.get('sl_order_id')
                    self.current_position['tp_order_id'] = conditional_results.get('tp_order_id')

                    # Log warnings if SL/TP failed to create
                    if not conditional_results.get('sl_success', False):
                        self.logger.warning("Stop loss order failed to create - position unprotected")
                    if not conditional_results.get('tp_success', False):
                        self.logger.warning("Take profit order failed to create - position may stay open")

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
                    self.logger.error(f"Position verification failed after retries. Expected: {side} {quantity}, Got: {verified_position.side if verified_position else 'None'} {verified_position.size if verified_position else 'N/A'}")
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
        """Calculate position size based on config and risk, accounting for trading fees."""
        balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
        leverage = self.config.get('LEVERAGE', 1)

        # Apply risk management percentage to balance first
        max_risk_pct = self.config.get('MAX_RISK_PERCENT', 1.0)
        max_position_value = balance * (max_risk_pct / 100.0)

        # Account for trading fees - estimate fee impact on position sizing
        # Since fees are deducted on exit, we need to ensure we can withstand the fee cost
        taker_fee = self.config.get('TAKER_FEE', 0.0006)  # 0.06% default
        # Conservative estimate: assume we might pay fees twice (entry + exit in worst case)
        fee_buffer = taker_fee * 2  # 0.12% total fee buffer

        # Adjust position value to account for fees
        effective_max_position_value = max_position_value * (1 - fee_buffer)

        # Calculate position value considering leverage
        # For futures: position_value = effective_max_position_value * leverage
        position_value = effective_max_position_value * leverage
        quantity = position_value / price

        # Apply volatility adjustment if enabled
        if self.config.get('VOLATILITY_ADJUSTED', True):
            try:
                ohlcv = await self.exchange.get_ohlcv(self.config['SYMBOL'], timeframe=60, limit=20)
                if len(ohlcv) >= 15:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    atr = self._calculate_atr(df, self.config.get('ATR_PERIOD', 14))
                    if atr > 0:
                        vol_adjustment = 1.0 / (1 + (atr / price) * 10)
                        quantity *= vol_adjustment
                        self.logger.info(f"Applied volatility adjustment: {vol_adjustment:.3f} (ATR: {atr:.2f})")
            except Exception as e:
                self.logger.warning(f"Failed to apply volatility adjustment: {e}")

        # Round to 4 decimal places for precision
        quantity = round(quantity, 4)

        # Enforce minimum position size
        min_size = 0.001 if 'BTC' in self.config['SYMBOL'] else 0.01  # Adjust for different symbols
        if quantity < min_size:
            quantity = min_size
            self.logger.info(f"Increased position size to minimum: {min_size}")

        self.app_logger.info(f"Position size: {quantity} {self.config['SYMBOL']} | Balance: ${balance:,.0f} | Max Risk: {max_risk_pct:.1f}% | Leverage: {leverage}x | Fee Buffer: {fee_buffer:.3f} | Price: ${price:,.1f}")
        return quantity

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr

    async def monitor_positions(self):
        """Monitor open positions for SL/TP status."""
        if not self.current_position:
            return

        try:
            symbol = self.config['SYMBOL']

            # Check SL/TP order status
            sl_order_id = self.current_position.get('sl_order_id')
            tp_order_id = self.current_position.get('tp_order_id')

            if sl_order_id:
                try:
                    sl_status = await self.exchange.get_order_status(symbol, sl_order_id)
                    if sl_status and sl_status.get('status') == 'filled':
                        self.logger.info("SL order filled - updating position state")
                        await self._update_position_from_exchange()
                except Exception as e:
                    self.logger.warning(f"Failed to check SL order {sl_order_id}: {e}")

            if tp_order_id:
                try:
                    tp_status = await self.exchange.get_order_status(symbol, tp_order_id)
                    if tp_status and tp_status.get('status') == 'filled':
                        self.logger.info("TP order filled - updating position state")
                        await self._update_position_from_exchange()
                except Exception as e:
                    self.logger.warning(f"Failed to check TP order {tp_order_id}: {e}")

        except Exception as e:
            self.logger.error(f"Error in position monitoring: {e}")

    async def _close_position(self, side: str, price: float, reason: str):
        """Close current position with validation."""
        if not self.current_position:
            return

        try:
            # Update position state before closure
            await self._update_position_from_exchange()

            symbol = self.config['SYMBOL']
            expected_quantity = self.current_position['quantity']
            expected_side = self.current_position['side']

            # Get the current position from exchange and validate it matches expected
            position = await self.exchange.get_pending_positions(symbol)
            if not position:
                self.logger.warning(f"No open position found for {symbol}")
                self.current_position = None
                return

            # Cancel SL/TP orders before closing position
            sl_order_id = self.current_position.get('sl_order_id')
            tp_order_id = self.current_position.get('tp_order_id')

            if sl_order_id:
                try:
                    cancelled = await self.exchange.cancel_order(symbol, sl_order_id)
                    if cancelled:
                        self.logger.info(f"Cancelled SL order {sl_order_id} before closing position")
                    else:
                        self.logger.warning(f"Failed to cancel SL order {sl_order_id}")
                except Exception as e:
                    self.logger.warning(f"Error cancelling SL order {sl_order_id}: {e}")

            if tp_order_id:
                try:
                    cancelled = await self.exchange.cancel_order(symbol, tp_order_id)
                    if cancelled:
                        self.logger.info(f"Cancelled TP order {tp_order_id} before closing position")
                    else:
                        self.logger.warning(f"Failed to cancel TP order {tp_order_id}")
                except Exception as e:
                    self.logger.warning(f"Error cancelling TP order {tp_order_id}: {e}")

            # Validate position but allow reasonable tolerance for live trading
            if position.side != expected_side:
                self.logger.error(f"CRITICAL: Position side mismatch - Expected: {expected_side}, Got: {position.side}. REFUSING TO CLOSE to prevent wrong position closure.")
                # Side mismatch is serious - don't close
                self.current_position = {
                    'side': position.side,
                    'quantity': position.size,
                    'entry_price': position.entry_price,
                    'timestamp': datetime.fromtimestamp(position.timestamp / 1000) if position.timestamp > 1e10 else datetime.fromtimestamp(position.timestamp)
                }
                return
                
            # For size validation, be more permissive in LIVE mode vs PAPER mode
            # Paper mode uses exact simulation, but live exchanges can have rounding
            if self.config.get('FORWARD_TESTING', False):
                # In paper mode - require exact match
                size_tolerance = 0.001  # Almost exact for paper trading simulation
            else:
                # In live trading - allow reasonable tolerance for exchange rounding
                size_tolerance = expected_quantity * 0.05  # 5% tolerance for live trading
            
            if abs(position.size - expected_quantity) > size_tolerance:
                self.logger.warning(f"Position size mismatch - Expected: {expected_quantity}, Got: {position.size}. Closing anyway with actual size to prevent stuck positions.")
                # Adjust to actual size and continue closing
                expected_quantity = position.size

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
