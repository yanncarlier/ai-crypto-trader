# config/settings.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingConfig:
    # Trading Configuration
    # Note: Values marked with [ENV] can be overridden in .env file
    CRYPTO: str = "Bitcoin"
    SYMBOL: str = "BTCUSDT"  # [ENV] Trading pair
    CURRENCY: str = "USDT"   # [ENV] Base currency
    CYCLE_MINUTES: int = 10  # [ENV] Analysis cycle length in minutes
    LEVERAGE: int = 2        # [ENV] Leverage (1-125x)
    MARGIN_MODE: str = "ISOLATED"  # [ENV] "ISOLATED" or "CROSS"
    POSITION_SIZE: str = "10%"     # [ENV] "10%", "50%", or "500" (fixed USDT)
    STOP_LOSS_PERCENT: Optional[int] = 10  # [ENV] Stop loss percentage
    TAKE_PROFIT_PERCENT: Optional[int] = None  # [ENV] Take profit percentage
    INITIAL_CAPITAL: float = 10_000.0  # [ENV] Initial trading capital
    TAKER_FEE: float = 0.0006  # [ENV] Exchange taker fee
    
    # Risk Management
    # [ENV] All risk management parameters can be overridden in .env
    MAX_POSITION_SIZE_PCT: float = 0.1      # Max position size as % of account
    DAILY_LOSS_LIMIT_PCT: float = 0.02      # Max daily loss as % of account
    MAX_DRAWDOWN_PCT: float = 0.05          # Max drawdown before stop trading
    MAX_HOLD_HOURS: int = 24                # Max hours to hold a position
    
    # Configuration
    FORWARD_TESTING: bool = False  # [ENV] Enable paper trading mode
    LLM_PROVIDER: str = "deepseek"  # [ENV] AI provider (xai, groq, openai, etc.)
    LLM_MODEL: str = "default"      # [ENV] Model to use with the provider
    LLM_TEMPERATURE: float = 0.2    # [ENV] AI temperature (0-1)
    LLM_MAX_TOKENS: int = 800       # [ENV] Max tokens for AI response
    EXCHANGE: str = "BITUNIX"       # [ENV] Exchange to use
    TEST_NET: bool = False          # [ENV] Use testnet if available

    @property
    def RUN_NAME(self) -> str:
        mode = "paper" if self.FORWARD_TESTING else "LIVE"
        return f"{mode}_{self.EXCHANGE}_{self.CRYPTO}_{self.SYMBOL}_{self.CYCLE_MINUTES}min_{self.LEVERAGE}x"