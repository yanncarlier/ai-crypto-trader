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
        self.risk_manager = RiskManager(self.config, self.exchange)
        self.current_position = None
        self.last_trade_time = None
        self.start_time = datetime.now()

    async def _update_position_from_exchange(self):
        """Update local position state from exchange."""
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
                'timestamp': timestamp,
                'stop_loss': self._calculate_stop_loss(position.side, position.entry_price),
                'take_profit': self._calculate_take_profit(position.side, position.entry_price)
            }
            self.last_trade_time = timestamp
        else:
            self.current_position = None
            self.last_trade_time = None

    async def run_cycle(self):
        """Run a single trading cycle: analyze market, get AI decision, execute trade if appropriate."""
        try:
            print("DEBUG: Starting trading cycle")

            # Update local position state from exchange
            await self._update_position_from_exchange()
            # Monitor current position for SL/TP/hold time
            await self.monitor_positions()

            # Fetch market data
            # print("DEBUG: About to fetch OHLCV")
            ohlcv = await self.exchange.get_ohlcv(
                self.config['SYMBOL'],
                timeframe=self.config['CYCLE_MINUTES'],
                limit=100
            )
            # print(f"DEBUG: OHLCV fetched, length: {len(ohlcv) if ohlcv else 0}")
            if not ohlcv or len(ohlcv) < 50:
                self.logger.warning("Insufficient market data")
                return

            # print("DEBUG: Creating DataFrame from OHLCV")
            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(
                df['timestamp'].astype(int), unit='ms')
            df.set_index('timestamp', inplace=True)
            # print("DEBUG: DataFrame created and indexed")

            # Calculate technical indicators
            # print("DEBUG: Calculating technical indicators")
            df['rsi'] = ta.rsi(df['close'], length=14)
            # print("DEBUG: RSI calculated")
            df['sma_20'] = ta.sma(df['close'], length=20)
            # print("DEBUG: SMA20 calculated")
            df['sma_50'] = ta.sma(df['close'], length=50)
            # print("DEBUG: SMA50 calculated")
            std20 = df['close'].rolling(window=20).std()
            df['bb_middle'] = df['sma_20']
            df['bb_upper'] = df['sma_20'] + (2 * std20)
            df['bb_lower'] = df['sma_20'] - (2 * std20)
            # print("DEBUG: Bollinger Bands calculated")
            df['ema_20'] = ta.ema(df['close'], length=20)
            # print("DEBUG: EMA20 calculated")
            macd = ta.macd(df['close'])
            df['MACD_hist'] = macd['MACDh_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            # print("DEBUG: MACD calculated")

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
            print(
                f"DEBUG: Indicators ready, candle price: {current_price}, live price: {live_price} ({price_diff_pct:+.2f}%), trend: {latest_indicators['trend']}")

            # Get AI outlook
            # print("DEBUG: Preparing for AI prompt")
            timestamp = datetime.now()
            minutes_elapsed = int((datetime.now() - self.start_time).total_seconds() // 60)
            try:
                account_balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
            except Exception as e:
                self.logger.warning(f"Failed to fetch account balance: {e}, using INITIAL_CAPITAL")
                account_balance = self.config['INITIAL_CAPITAL']
            equity = account_balance  # TODO: include unrealized PnL if any
            open_positions = [
                self.current_position] if self.current_position else []
            price_history_short = df.tail(20).reset_index().to_dict('records')
            price_history_long = df.tail(50).reset_index().to_dict('records')
            indicators = latest_indicators
            predictive_signals = {}
            # print("DEBUG: Calling build_prompt")
            prompt = build_prompt(
                timestamp, minutes_elapsed, account_balance, equity, open_positions,
                price_history_short, price_history_long, indicators, predictive_signals, self.config
            )
            # print(f"DEBUG: Prompt built, length: {len(prompt)}")

            # print("DEBUG: Calling send_request")
            outlook = await send_request(prompt, self.config)
            # print(f"DEBUG: AI outlook received: {outlook}")
            save_response(outlook, self.config['RUN_NAME'])

            # Parse AI outlook
            # print("DEBUG: Parsing AI outlook")
            ai_decision = self._parse_ai_outlook(outlook)
            print(f"DEBUG: AI decision: {ai_decision}")

            # Check risk management
            # print("DEBUG: Checking risk management")
            if not await self.risk_manager.can_trade(ai_decision, current_price, self.current_position):
                self.logger.info(
                    f"Action: {ai_decision['action']}, confidence: {ai_decision['confidence']:.2f}")
                return

            # Execute trade
            if ai_decision['action'] in ['BUY', 'SELL', 'CLOSE_POSITION']:
                print("DEBUG: Executing trade")
                await self._execute_trade(ai_decision, current_price)

            self.logger.info(
                f"Cycle completed. AI Decision: {ai_decision['action']}")
            print("DEBUG: Cycle completed successfully")

        except Exception as e:
            print(f"DEBUG: Exception in run_cycle: {type(e).__name__}: {e}")
            import traceback
            print("DEBUG: Full traceback:")
            traceback.print_exc()
            self.logger.error(f"Error in run_cycle: {e}")
            raise

    def _parse_ai_outlook(self, outlook: AIOutlook) -> Dict[str, Any]:
        """Parse AI outlook into actionable decision."""
        # Use the action provided by the AI, or map interpretation to action if action is not provided
        if outlook.action and outlook.action in ['BUY', 'SELL', 'CLOSE_POSITION', 'HOLD', 'NO_TRADE']:
            action = outlook.action
        else:
            # Map interpretation to action, default to HOLD if unknown
            if outlook.interpretation == 'Bullish':
                action = 'BUY'
            elif outlook.interpretation == 'Bearish':
                action = 'SELL'
            else:
                action = 'HOLD'
        
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
                self.logger.warning(
                    "Calculated position size is zero or negative")
                return

            # Place order using open_position with stop loss and take profit
            sl_pct = self.config['STOP_LOSS_PERCENT']
            tp_pct = self.config['TAKE_PROFIT_PERCENT']
            
            order = await self.exchange.open_position(
                symbol=symbol,
                side=side,
                size=str(quantity),
                sl_pct=sl_pct,
                tp_pct=tp_pct
            )

            if order:
                self.current_position = {
                    'side': side,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'timestamp': datetime.now(),
                    'stop_loss': self._calculate_stop_loss(side, current_price),
                    'take_profit': self._calculate_take_profit(side, current_price)
                }
                self.last_trade_time = datetime.now()
                self.logger.info(
                    f"Trade executed: {side} {quantity} {symbol} at {current_price} with SL: {sl_pct}% and TP: {tp_pct}%")
                balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
                trade = {
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'side': side,
                    'quantity': quantity,
                    'price': current_price,
                    'pnl': 0.0,
                    'balance': balance
                }
                self.risk_manager.update_trade_history(trade)
            else:
                self.logger.error(f"Order failed: {order}")

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")

    async def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on config and risk."""
        try:
            balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
        except Exception as e:
            self.logger.warning(f"Failed to fetch account balance for position sizing: {e}, using INITIAL_CAPITAL")
            balance = self.config['INITIAL_CAPITAL']  # Use actual balance from env
        max_size_pct = self.config['MAX_POSITION_SIZE_PCT']  # Already a decimal from config
        position_value = balance * max_size_pct
        quantity = position_value / price
        
        # Ensure minimum position size and avoid tiny positions
        min_quantity = self.config['MIN_POSITION_SIZE']
        quantity = max(quantity, min_quantity)
        
        # Round to 4 decimal places for precision, but ensure minimum
        quantity = round(quantity, 4)
        if quantity < min_quantity:
            quantity = min_quantity
            
        self.logger.info(f"Calculated position size: {quantity} {self.config['SYMBOL']}, balance: {balance} USDT, position_value: {position_value} USDT, price: {price}")
        return quantity

    def _calculate_stop_loss(self, side: str, entry_price: float) -> float:
        """Calculate stop loss price."""
        sl_pct = self.config['STOP_LOSS_PERCENT'] / 100
        if side == 'BUY':
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def _calculate_take_profit(self, side: str, entry_price: float) -> float:
        """Calculate take profit price."""
        tp_pct = self.config['TAKE_PROFIT_PERCENT'] / 100
        if side == 'BUY':
            return entry_price * (1 + tp_pct)
        else:
            return entry_price * (1 - tp_pct)

    async def monitor_positions(self):
        """Monitor open positions and manage exits."""
        if not self.current_position:
            return

        try:
            current_price = await self.exchange.get_ticker(self.config['SYMBOL'])
            pos = self.current_position
            side = pos['side']

            # Check stop loss / take profit
            if side == 'BUY':
                if current_price <= pos['stop_loss']:
                    await self._close_position('SELL', current_price, "Stop Loss")
                elif current_price >= pos['take_profit']:
                    await self._close_position('SELL', current_price, "Take Profit")
            else:
                if current_price >= pos['stop_loss']:
                    await self._close_position('BUY', current_price, "Stop Loss")
                elif current_price <= pos['take_profit']:
                    await self._close_position('BUY', current_price, "Take Profit")

            # Check max hold time
            if self.last_trade_time:
                if (datetime.now() - self.last_trade_time) > timedelta(hours=self.config['MAX_HOLD_HOURS']):
                    # Determine correct side to close: opposite of current position side
                    close_side = 'SELL' if side == 'BUY' else 'BUY'
                    await self._close_position(close_side, current_price, "Max Hold Time")

        except Exception as e:
            self.logger.error(f"Error monitoring position: {e}")

    async def _close_position(self, side: str, price: float, reason: str):
        """Close current position."""
        if not self.current_position:
            return

        try:
            symbol = self.config['SYMBOL']
            quantity = self.current_position['quantity']

            # Get the current position from exchange
            position = await self.exchange.get_pending_positions(symbol)
            if not position:
                self.logger.warning(f"No open position found for {symbol}")
                self.current_position = None
                return

            # Use the proper close_position method
            result = await self.exchange.close_position(position, reason)
            pnl = result.get('pnl', 0.0)

            if result.get('order'):
                self.logger.info(f"Position closed: {reason} at {price}")
                self.current_position = None
                self.risk_manager.update_daily_pnl(pnl)
                # Fetch current balance for trade record
                try:
                    balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
                except Exception as e:
                    self.logger.warning(f"Failed to fetch account balance for trade record: {e}")
                    balance = self.config['INITIAL_CAPITAL']
                trade = {
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'pnl': pnl,
                    'balance': balance
                }
                self.risk_manager.update_trade_history(trade)
            else:
                self.logger.error(f"Failed to close position")

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
