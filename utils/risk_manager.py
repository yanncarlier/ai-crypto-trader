import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import time

@dataclass
class RiskParameters:
    max_position_size_pct: float = 0.1  # 10% of account per trade
    daily_loss_limit_pct: float = 0.02  # 2% daily loss limit
    max_drawdown_pct: float = 0.05  # 5% max drawdown per trade
    max_hold_period_hours: int = 24  # Close positions after 24 hours
    volatility_adjusted: bool = True  # Adjust position size based on volatility
    atr_period: int = 14  # ATR period for volatility calculation
    max_leverage: int = 5  # Maximum allowed leverage
    min_liquidity: float = 1000000  # Minimum 24h volume in USDT

class RiskManager:
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.risk_params = RiskParameters()
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        
    async def calculate_position_size(self, symbol: str, price: float, position_size: float) -> Tuple[float, Dict]:
        """Calculate position size based on risk parameters"""
        try:
            account_info = await self.exchange.get_account_summary(self.config.CURRENCY, symbol)
            balance = account_info['balance']

            # Get market data for volatility calculation
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1d', limit=self.risk_params.atr_period + 1)
            if len(ohlcv) < 2:
                raise ValueError("Not enough data for volatility calculation")

            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate position size based on risk parameters
            max_position_value = balance * self.risk_params.max_position_size_pct

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
                    daily_pnl) / (balance * self.risk_params.daily_loss_limit_pct)
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
            balance = await self.exchange.get_account_balance(self.config.CURRENCY)
            fallback_size = (balance * 0.01) / price
            return min(position_size, fallback_size), {'error': str(e)}
    
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
        trade_date = pd.to_datetime(trade['timestamp'], unit='ms').strftime('%Y-%m-%d')
        self.daily_pnl[trade_date] = self.daily_pnl.get(trade_date, 0) + trade.get('pnl', 0)
        
    async def check_risk_limits(self, symbol: str) -> Tuple[bool, str]:
        """Check if trading should be paused due to risk limits"""
        try:
            balance = await self.exchange.get_account_balance(self.config.CURRENCY)
            
            # Check daily loss limit
            today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
            if today in self.daily_pnl and self.daily_pnl[today] < 0:
                max_daily_loss = self.risk_params.daily_loss_limit_pct * self.config.INITIAL_CAPITAL
                if abs(self.daily_pnl[today]) >= max_daily_loss:
                    return False, f"Daily loss limit reached: {self.daily_pnl[today]:.2f} {self.config.CURRENCY}"
            
            # Check max drawdown
            if self.trade_history:
                peak = max(t.get('balance', 0) for t in self.trade_history + [{'balance': balance}])
                drawdown = (peak - balance) / peak if peak > 0 else 0
                if drawdown > self.risk_params.max_drawdown_pct:
                    return False, f"Max drawdown exceeded: {drawdown*100:.2f}%"
            
            return True, "OK"
            
        except Exception as e:
            logging.error(f"Error checking risk limits: {e}")
            return False, f"Risk check error: {str(e)}"
    
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
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if total_trades > winning_trades else 0
            profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else float('inf')
            
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
