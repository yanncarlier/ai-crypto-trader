import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import time


@dataclass
class RiskParameters:
    volatility_adjusted: bool
    atr_period: int
    max_leverage: int
    max_hold_period_hours: int = 24


class RiskManager:
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.risk_params = RiskParameters(
            volatility_adjusted=config.get('VOLATILITY_ADJUSTED', True),
            atr_period=config.get('ATR_PERIOD', 14),
            max_leverage=config['LEVERAGE']
        )
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}

    async def calculate_position_size(self, symbol: str, price: float, position_size: float) -> Tuple[float, Dict]:
        """Calculate position size based on risk parameters"""
        try:
            account_info = await self.exchange.get_account_summary(self.config['CURRENCY'], symbol)
            if account_info is None:
                raise ValueError("Failed to fetch account info")
            balance = account_info['balance']

            # Get market data for volatility calculation with fallback
            try:
                ohlcv = await self.exchange.get_ohlcv(symbol, timeframe=60, limit=self.risk_params.atr_period + 1)
                if len(ohlcv) < 2:
                    logging.warning("Insufficient data for volatility calculation, disabling volatility adjustment")
                    self.risk_params.volatility_adjusted = False
            except Exception as market_err:
                logging.warning(f"Failed to fetch market data for volatility: {market_err}, proceeding without volatility adjustment")
                self.risk_params.volatility_adjusted = False
                ohlcv = []

            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate position size based on risk parameters
            max_position_value = balance  # Use full balance

            if self.risk_params.volatility_adjusted:
                # Use ATR for volatility-based position sizing
                atr = self._calculate_atr(df)
                if atr > 0:
                    # Reduce position size if volatility is high
                    vol_adjustment = 1.0 / (1 + (atr / price) * 10)
                    max_position_value *= vol_adjustment

            # Calculate number of contracts and apply limits
            contract_size = max_position_value / price
            final_size = min(position_size, contract_size)

            return final_size, {
                'max_position_value': max_position_value,
                'volatility_adjusted': self.risk_params.volatility_adjusted
            }

        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            # Fallback to conservative position sizing on error
            try:
                balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
                # Use 10% of balance as conservative fallback
                fallback_size = (balance * 0.10) / price
                # Ensure fallback is reasonable (not too large)
                max_fallback = position_size * 0.5  # Max 50% of requested size
                safe_fallback = min(fallback_size, max_fallback)
                logging.warning(f"Using fallback position size: {safe_fallback}")
                return safe_fallback, {'error': str(e), 'fallback': True}
            except Exception as fallback_error:
                logging.error(f"Fallback position sizing also failed: {fallback_error}")
                return 0.0, {'error': f"{str(e)} | Fallback failed: {str(fallback_error)}"}

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.risk_params.atr_period).mean().iloc[-1]
        return atr

    def update_trade_history(self, trade: Dict):
        """Update trade history for performance analysis"""
        self.trade_history.append(trade)

        # Update daily PnL
        trade_date = pd.to_datetime(
            trade['timestamp'], unit='ms').strftime('%Y-%m-%d')
        self.daily_pnl[trade_date] = self.daily_pnl.get(
            trade_date, 0) + trade.get('pnl', 0)

    async def check_risk_limits(self, symbol: str) -> Tuple[bool, str]:
        """Check if trading should be paused due to risk limits"""
        return True, "OK"

    async def can_trade(self, decision: Dict, current_price: float, current_position: Optional[Dict]) -> bool:
        """Check if a trade can be executed based on risk rules."""
        try:
            # Allow closing positions even if already in one
            if decision.get('action') == 'CLOSE_POSITION':
                return True

            # Check if already in position (only block new positions)
            if current_position and decision.get('action') in ['BUY', 'SELL']:
                logging.info("Already in position - blocking new trade")
                return False

            # Check overall risk limits
            can_proceed, reason = await self.check_risk_limits(decision.get('symbol', self.config['SYMBOL']))
            if not can_proceed:
                logging.warning(f"Risk limit violated: {reason}")
                return False

            # Check confidence threshold
            confidence = decision.get('confidence', 0)
            min_confidence = self.config.get('CONFIDENCE_THRESHOLD', 0.7)
            if confidence < min_confidence:
                logging.info(f"AI confidence {confidence:.2f} below threshold {min_confidence:.2f}")
                return False

            # Check position size feasibility
            balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
            leverage = self.config.get('LEVERAGE', 1)
            max_risk_pct = self.config.get('MAX_RISK_PERCENT', 1.0)

            # Calculate maximum position value with risk limits
            max_position_value = balance * (max_risk_pct / 100.0) * leverage
            proposed_size = max_position_value / current_price

            if proposed_size <= 0:
                logging.warning("Calculated position size is zero or negative")
                return False

            # Additional check: ensure position size doesn't exceed exchange limits
            if proposed_size * current_price > balance * leverage:
                logging.warning("Position size exceeds leverage limits")
                return False

            return True
        except Exception as e:
            logging.error(f"Error in can_trade: {e}")
            return False

    def update_daily_pnl(self, pnl: float):
        """Update daily PnL tracking."""
        today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl
        logging.info(
            f"Updated daily PnL: {pnl}, total for today: {self.daily_pnl[today]}")

    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        if not self.trade_history:
            return {}

        try:
            df = pd.DataFrame(self.trade_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate metrics
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = df[df['pnl'] > 0]['pnl'].mean(
            ) if winning_trades > 0 else 0
            avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()
                           ) if total_trades > winning_trades else 0
            profit_factor = (win_rate * avg_win) / ((1 - win_rate)
                                                    * avg_loss) if avg_loss > 0 else float('inf')

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_pnl': df['pnl'].sum(),
                'max_drawdown': self._calculate_max_drawdown(df),
                'daily_pnl': self.daily_pnl
            }
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        df['cum_max'] = df['balance'].cummax()
        df['drawdown'] = (df['cum_max'] - df['balance']) / df['cum_max']
        return df['drawdown'].max() * 100  # Return as percentage
