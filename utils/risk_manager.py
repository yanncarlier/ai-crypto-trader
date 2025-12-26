import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import time


@dataclass
class RiskParameters:
    max_position_size_pct: float
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    max_hold_period_hours: int
    volatility_adjusted: bool
    atr_period: int
    max_leverage: int
    min_liquidity: float


class RiskManager:
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.risk_params = RiskParameters(
            max_position_size_pct=config['MAX_POSITION_SIZE_PCT'] / 100,
            daily_loss_limit_pct=config['DAILY_LOSS_LIMIT_PCT'] / 100,
            max_drawdown_pct=config['MAX_DRAWDOWN_PCT'] / 100,
            max_hold_period_hours=config['MAX_HOLD_HOURS'],
            volatility_adjusted=config.get('VOLATILITY_ADJUSTED', True),
            atr_period=config.get('ATR_PERIOD', 14),
            max_leverage=config['LEVERAGE'],
            min_liquidity=config.get('MIN_LIQUIDITY', 1000000)
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
            max_position_value = balance * self.config['MAX_POSITION_SIZE_PCT']

            if self.risk_params.volatility_adjusted:
                # Use ATR for volatility-based position sizing
                atr = self._calculate_atr(df)
                if atr > 0:
                    # Reduce position size if volatility is high
                    vol_adjustment = 1.0 / (1 + (atr / price) * 10)
                    max_position_value *= vol_adjustment

            # Check daily loss limit
            today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
            daily_pnl = self.daily_pnl.get(today, 0)
            if daily_pnl < 0:
                # Reduce position size if approaching daily loss limit
                loss_ratio = abs(
                    daily_pnl) / (balance * self.config['DAILY_LOSS_LIMIT_PCT'])
                max_position_value *= max(0, 1 - loss_ratio)

            # Calculate number of contracts and apply limits
            contract_size = max_position_value / price
            final_size = min(position_size, contract_size)

            return final_size, {
                'max_position_value': max_position_value,
                'volatility_adjusted': self.risk_params.volatility_adjusted,
                'daily_pnl_impact': daily_pnl
            }

        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            # Fallback to 1% of balance on error
            try:
                balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
                fallback_size = (balance * 0.01) / price
                return min(position_size, fallback_size), {'error': str(e)}
            except:
                return 0.0, {'error': str(e)}

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
        try:
            balance = await self.exchange.get_account_balance(self.config['CURRENCY'])

            # Check daily loss limit
            today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
            if today in self.daily_pnl and self.daily_pnl[today] < 0:
                max_daily_loss = balance * self.config['DAILY_LOSS_LIMIT_PCT']
                if abs(self.daily_pnl[today]) >= max_daily_loss:
                    return False, f"Daily loss limit reached: {self.daily_pnl[today]:.2f} {self.config['CURRENCY']}"

            # Check max drawdown
            if self.trade_history:
                peak = max(t.get('balance', 0)
                           for t in self.trade_history + [{'balance': balance}])
                drawdown = (peak - balance) / peak if peak > 0 else 0
                if drawdown > self.config['MAX_DRAWDOWN_PCT']:
                    return False, f"Max drawdown exceeded: {drawdown*100:.2f}%"

            return True, "OK"

        except Exception as e:
            logging.error(f"Error checking risk limits: {e}")
            return False, f"Risk check error: {str(e)}"

    async def can_trade(self, decision: Dict, current_price: float, current_position: Optional[Dict]) -> bool:
        """Check if a trade can be executed based on risk rules."""
        try:
            # Check if already in position
            if current_position:
                return False  # Only one position at a time

            # Check overall risk limits
            can_proceed, reason = await self.check_risk_limits(decision.get('symbol', self.config['SYMBOL']))
            if not can_proceed:
                logging.warning(f"Risk limit violated: {reason}")
                return False

            # Check confidence threshold
            confidence = decision.get('confidence', 0)
            if confidence < self.config.get('MIN_CONFIDENCE', 0.5):
                logging.info("AI confidence too low for trade")
                return False

            # Check position size feasibility
            balance = await self.exchange.get_account_balance(self.config['CURRENCY'])
            proposed_size = (balance * self.config['MAX_POSITION_SIZE_PCT']) / current_price
            adjusted_size, details = await self.calculate_position_size(self.config['SYMBOL'], current_price, proposed_size)
            if adjusted_size <= 0:
                logging.warning("Adjusted position size is zero")
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
