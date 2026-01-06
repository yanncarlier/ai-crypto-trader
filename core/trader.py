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
        """Execute a trade based on AI decision with atomic verification and rollback."""
        try:
            symbol = self.config['SYMBOL']
            side = decision['action']
            app_logger = self.app_logger
            
            app_logger.info(f"Starting atomic trade execution: {side} position attempt")
            
            # Atomically check and update position state first
            existing_position = await self.exchange.get_pending_positions(symbol)
            if existing_position:
                # Handle position reversals - if AI wants opposite direction, close existing first
                if ((side == 'BUY' and existing_position.side == 'SELL') or
                    (side == 'SELL' and existing_position.side == 'BUY')):
                    app_logger.info(f"Position reversal detected: closing {existing_position.side} to open {side}")
                    position_dict = {
                        'side': existing_position.side,
                        'quantity': existing_position.size,
                        'entry_price': existing_position.entry_price,
                        'sl_order_id': getattr(existing_position, 'sl_order_id', None),
                        'tp_order_id': getattr(existing_position, 'tp_order_id', None)
                    }
                    await self._close_position(position_dict['side'], await self.exchange.get_current_price(symbol), "Position reversal")
                    # Re-verify position is actually closed before proceeding
                    closed_pos = await self.exchange.get_pending_positions(symbol)
                    if closed_pos:
                        app_logger.error("Position reversal failed - position still exists, aborting new trade")
                        return
                else:
                    # Same direction - already in position, prevent duplicate
                    self.logger.warning(f"AI requested {side} but already in {existing_position.side} position - blocking duplicate")
                    self.current_position = {
                        'side': existing_position.side,
                        'quantity': existing_position.size,
                        'entry_price': existing_position.entry_price,
                        'timestamp': datetime.fromtimestamp(existing_position.timestamp / 1000) if position.timestamp >= 1e10 else datetime.fromtimestamp(existing_position.timestamp)
                    }
                    return

            # Calculate position size with validation
            quantity = await self._calculate_position_size(current_price)
            if quantity <= 0:
                self.logger.warning("Calculated position size is zero or negative, aborting trade")
                return

            # Atomically execute trade with comprehensive verification
            await self._atomic_trade_execution(side, quantity, symbol, current_price)

        except Exception as e:
            # Comprehensive rollback on any failure
            self.logger.error(f"Trade execution failed with exception: {e}")
            await self._rollback_failed_trade()
            raise

    async def _atomic_trade_execution(self, side: str, quantity: float, symbol: str, current_price: float):
        """Atomically execute and verify the trade with proper rollback mechanisms."""
        original_position = self.current_position.copy() if self.current_position else None
        temp_position = None
        
        try:
            app_logger = self.app_logger
            app_logger.info(f"Attempting atomic position open: {side} {quantity:.4f} {symbol}")
            
            # Create a temporary lock-like mechanism for this atomic operation
            # This ensures no concurrent operations corrupt our state
            order_result = None
            
            # Retry mechanism with exponential backoff for order placement
            for attempt in range(3):
                try:
                    order_result = await self.exchange.open_position(
                        symbol=symbol,
                        side=side,
                        size=str(quantity)
                    )
                    
                    if order_result and order_result.get('orderId'):
                        app_logger.info(f"Order placed successfully: {order_result['orderId']}")
                        break
                    else:
                        self.logger.warning(f"Order attempt {attempt + 1} returned no order ID")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
                except Exception as order_error:
                    self.logger.error(f"Order attempt {attempt + 1} failed: {order_error}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        self.logger.error("All order attempts failed, triggering rollback")
                        await self._rollback_failed_trade()
                        return

            if not order_result or not order_result.get('orderId'):
                self.logger.error("No order result or order ID available")
                await self._rollback_failed_trade()
                return

            # CRITICAL VERIFICATION PHASE - Atomic verification sequence
            order_id = order_result['orderId']
            verified_position = None
            
            # Comprehensive verification with timeout management
            max_verification_time = 60  # 60 seconds maximum for verification
            verification_start = datetime.now()
            
            while (datetime.now() - verification_start).seconds < max_verification_time:
                try:
                    # Step 1: Verify order is processed by exchange
                    order_status = await self.exchange.get_order_status(symbol, order_id)
                    if order_status:
                        order_state = order_status.get('status')
                        self.logger.info(f"Order {order_id} status: {order_state}")
                        
                        if order_state == 'filled':
                            app_logger.info(f"Order {order_id} confirmed as filled")
                            break
                        elif order_state in ['canceled', 'rejected']:
                            self.logger.error(f"Order {order_id} was {order_state}, triggering rollback")
                            await self._rollback_failed_trade()
                            return
                    
                    # Step 2: Check if a position actually exists on the exchange
                    verified_position = await self.exchange.get_pending_positions(symbol)
                    if verified_position and verified_position.side == side:
                        size_tolerance = quantity * 0.03  # 3% size tolerance
                        if abs(verified_position.size - quantity) <= size_tolerance:
                            app_logger.info(f"Position verified: {verified_position.side} {verified_position.size:.4f} @ ${verified_position.entry_price:,.2f}")
                            break
                    
                    await asyncio.sleep(5)  # Wait before retry
                    
                except Exception as verify_error:
                    self.logger.warning(f"Verification attempt failed: {verify_error}")
                    await asyncio.sleep(3)
                    continue

            # FINAL VALIDATION DECISION
            if not verified_position:
                # Final check after timeout
                final_check = await self.exchange.get_pending_positions(symbol)
                if final_check and final_check.side == side:
                    size_tolerance = quantity * 0.05  # 5% final tolerance
                    if abs(final_check.size - quantity) <= size_tolerance:
                        verified_position = final_check
                        app_logger.warning("Position found during final check, proceeding with warning")
                    else:
                        self.logger.error(f"Position size mismatch: expected {quantity}, got {final_check.size}")
                        await self._rollback_failed_trade()
                        return
                else:
                    self.logger.error("Position verification failed after timeout")
                    await self._rollback_failed_trade()
                    return

            # Create temporary position state for SL/TP setup
            temp_position = {
                'side': side,
                'quantity': quantity,
                'entry_price': verified_position.entry_price,
                'timestamp': datetime.fromtimestamp(verified_position.timestamp / 1000) if verified_position.timestamp >= 1e10 else datetime.fromtimestamp(verified_position.timestamp),
                'position_id': verified_position.positionId
            }

            # Setup SL/TP orders with comprehensive validation
            sl_pct = self.config.get('STOP_LOSS_PERCENT', 2.0)
            tp_pct = self.config.get('TAKE_PROFIT_PERCENT', 4.0)
            
            if sl_pct <= 0 or tp_pct <= 0:
                self.logger.error("Invalid SL/TP percentages, aborting trade setup")
                await self._rollback_failed_trade()
                return

            conditional_results = await self.exchange._create_conditional_orders(
                symbol, quantity, side, verified_position.entry_price, sl_pct, tp_pct
            )

            # Validate SL/TP creation was successful
            sl_success = conditional_results.get('sl_success', False)
            tp_success = conditional_results.get('tp_success', False)
            
            if not sl_success:
                self.logger.warning("Stop loss order failed to create - position opened but unprotected")
            if not tp_success:
                self.logger.warning("Take profit order failed to create - position may stay open longer")

            # Final atomic commit - update local state only after everything succeeded
            self.current_position = temp_position.copy()
            self.current_position.update({
                'sl_order_id': conditional_results.get('sl_order_id'),
                'tp_order_id': conditional_results.get('tp_order_id'),
                'order_id': order_id
            })
            
            self.last_trade_time = datetime.now()
            
            # Final success logging
            app_logger.info(f"✅ ATOMIC TRADE SUCCESS: {side} {quantity:.4f} {symbol} @ ${verified_position.entry_price:,.2f}")
            app_logger.info(f"📊 SL: {conditional_results.get('sl_order_id') or 'FAILED'} | TP: {conditional_results.get('tp_order_id') or 'FAILED'}")
            
            # Record trade in history
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

        except Exception as atomic_error:
            self.logger.error(f"Atomic trade execution failed: {atomic_error}")
            # Restore original state on any failure
            if original_position:
                self.current_position = original_position.copy()
            else:
                self.current_position = None
            # Clean up any partially created orders
            if temp_position and temp_position.get('order_id'):
                try:
                    await self.exchange.cancel_order(symbol, temp_position['order_id'])
                except:
                    pass  # Best effort cleanup
            if temp_position and temp_position.get('sl_order_id'):
                try:
                    await self.exchange.cancel_order(symbol, temp_position['sl_order_id'])
                except:
                    pass
            if temp_position and temp_position.get('tp_order_id'):
                try:
                    await self.exchange.cancel_order(symbol, temp_position['tp_order_id'])
                except:
                    pass
            raise

    async def _rollback_failed_trade(self):
        """Comprehensive rollback mechanism for failed trades."""
        try:
            symbol = self.config['SYMBOL']
            self.logger.warning("🔄 Executing trade rollback procedure")
            
            # Clear local position state since trade failed
            self.current_position = None
            self.last_trade_time = None
            
            # Verify exchange state is clean (no orphaned orders)
            try:
                open_orders = await self.exchange.get_open_orders(symbol)
                for order in open_orders:
                    try:
                        await self.exchange.cancel_order(symbol, order.get('orderId'))
                        self.logger.info(f"Cancelled orphaned order: {order.get('orderId')}")
                    except:
                        pass
            except:
                pass  # Best effort cleanup
        except Exception as rollback_error:
            self.logger.error(f"Rollback procedure encountered error: {rollback_error}")
            # Even if rollback partially fails, ensure we're in a clean state
            self.current_position = None
            self.last_trade_time = None

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
        """Close current position with atomic validation and rollback on failure."""
        if not self.current_position:
            return

        original_position = self.current_position.copy()
        symbol = self.config['SYMBOL']
        
        try:
            app_logger = self.app_logger
            app_logger.info(f"Starting atomic position closure: {reason} at ${price:,.1f}")
            
            # CRITICAL: Verify position exists before attempting to close
            current_exchange_position = await self.exchange.get_pending_positions(symbol)
            if not current_exchange_position:
                app_logger.warning(f"No position found on exchange for {symbol}")
                self.current_position = None
                return
                
            # Validate position details match what we expect
            expected_side = original_position['side']
            expected_quantity = original_position['quantity']
            
            if current_exchange_position.side != expected_side:
                app_logger.error(f"CRITICAL: Position side mismatch - Expected: {expected_side}, Got: {current_exchange_position.side}")
                await self._recover_position_state_mismatch(current_exchange_position)
                return
            
            # Use actual size from exchange for closing (handles exchange rounding)
            close_quantity = current_exchange_position.size
            
            app_logger.info(f"Closing {current_exchange_position.side} position: {close_quantity:.4f} {symbol} @ ${current_exchange_position.entry_price:,.2f}")

            # Create atomic operation with rollback capability
            close_result = await self._atomic_position_close(
                position=current_exchange_position,
                expected_quantity=close_quantity,
                price=price,
                reason=reason
            )
            
            if close_result.get('success', False):
                pnl = close_result.get('pnl', 0.0)
                app_logger.info(f"✅ POSITION CLOSED SUCCESSFULLY: {reason} | PnL: ${pnl:+.2f}")
                
                # Only update local state after confirmed success on exchange
                self.current_position = None
                self.risk_manager.update_daily_pnl(pnl)
                
                # Record trade for history
                balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
                trade = {
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'side': f'CLOSE_{expected_side}',
                    'quantity': close_quantity,
                    'price': price,
                    'pnl': pnl,
                    'balance': balance
                }
                self.risk_manager.update_trade_history(trade)
            else:
                app_logger.error(f"Position closure failed: {close_result.get('error', 'Unknown error')}")
                
                # Attempt recovery on closure failure
                await self._recover_position_state_mismatch(current_exchange_position)
                return

        except Exception as e:
            self.logger.error(f"Position closure failed with exception: {e}")
            # Restore original state on any failure
            self.current_position = original_position
            await self._recover_position_state_mismatch(await self.exchange.get_pending_positions(symbol))
            raise

    async def _atomic_position_close(self, position, expected_quantity, price, reason):
        """Atomically close position with comprehensive validation and cleanup."""
        result = {'success': False, 'pnl': 0.0, 'error': None}
        
        try:
            symbol = self.config['SYMBOL']
            
            # Step 1: Cancel any existing SL/TP orders first
            cancelled_orders = []
            failed_cancellations = []
            
            sl_order_id = self.current_position.get('sl_order_id')
            if sl_order_id:
                try:
                    cancelled = await self.exchange.cancel_order(symbol, sl_order_id)
                    if cancelled:
                        self.logger.info(f"Cancelled SL order {sl_order_id}")
                        cancelled_orders.append(sl_order_id)
                    else:
                        self.logger.warning(f"Failed to cancel SL order {sl_order_id}")
                        failed_cancellations.append(f"SL-{sl_order_id}")
                except Exception as e:
                    self.logger.warning(f"Error cancelling SL order: {e}")
                    failed_cancellations.append(f"SL-{sl_order_id}")
            
            tp_order_id = self.current_position.get('tp_order_id')
            if tp_order_id:
                try:
                    cancelled = await self.exchange.cancel_order(symbol, tp_order_id)
                    if cancelled:
                        self.logger.info(f"Cancelled TP order {tp_order_id}")
                        cancelled_orders.append(tp_order_id)
                    else:
                        self.logger.warning(f"Failed to cancel TP order {tp_order_id}")
                        failed_cancellations.append(f"TP-{tp_order_id}")
                except Exception as e:
                    self.logger.warning(f"Error cancelling TP order: {e}")
                    failed_cancellations.append(f"TP-{tp_order_id}")

            # Step 2: Close the position on the exchange
            close_result = await self.exchange.close_position(position, reason)
            
            if not close_result or not close_result.get('order'):
                result['error'] = "Exchange close operation failed - no order returned"
                return result
            
            # Step 3: Verify the position is actually gone
            max_verification_attempts = 5
            
            for attempt in range(max_verification_attempts):
                try:
                    await asyncio.sleep(1 + attempt)  # Increasing delays: 1s, 2s, 3s, 4s, 5s
                    
                    post_close_position = await self.exchange.get_pending_positions(symbol)
                    if not post_close_position:
                        # Success - position is gone
                        pnl = close_result.get('pnl', 0.0)
                        result.update({
                            'success': True,
                            'pnl': pnl,
                            'cancelled_orders': cancelled_orders,
                            'failed_cancellations': failed_cancellations
                        })
                        
                        self.app_logger.info(
                            f"Position closed: {reason} at ${price:,.1f}" +
                            (f" | PnL: ${pnl:+.2f}" if pnl != 0.0 else "")
                        )
                        
                        return result
                    else:
                        self.logger.warning(f"Position still exists after close attempt {attempt + 1}")
                        
                except Exception as verify_error:
                    self.logger.error(f"Position verification failed on attempt {attempt + 1}: {verify_error}")

            # If we get here, position verification failed after all attempts
            result['error'] = "Position still exists on exchange after close operation"
            
        except Exception as e:
            self.logger.error(f"Atomic position close failed: {e}")
            result['error'] = f"Position closure error: {str(e)}"
        
        return result

    async def _recover_position_state_mismatch(self, actual_position):
        """Recover from position state mismatches by synchronizing with exchange."""
        try:
            app_logger = self.app_logger
            
            if actual_position:
                # Update local state to match exchange reality
                updated_state = {
                    'side': actual_position.side,
                    'quantity': actual_position.size,
                    'entry_price': actual_position.entry_price,
                    'timestamp': datetime.fromtimestamp(actual_position.timestamp / 1000) if actual_position.timestamp >= 1e10 else datetime.fromtimestamp(actual_position.timestamp),
                    'position_id': actual_position.positionId
                }
                
                self.current_position = updated_state
                app_logger.warning(
                    f"Position state synchronized with exchange: {updated_state['side']} {updated_state['quantity']:.4f} @ ${updated_state['entry_price']:.2f}"
                )
            else:
                # No position on exchange, clear local state
                app_logger.info("No position found on exchange - local state cleared")
                self.current_position = None
                
            self.last_trade_time = None
            
        except Exception as recovery_error:
            self.logger.error(f"Position recovery failed: {recovery_error}")
            # Ensure we're not in a corrupted state
            self.current_position = None
            self.last_trade_time = None
