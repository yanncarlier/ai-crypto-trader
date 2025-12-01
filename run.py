# run.py
from core.trader import TradingBot
from exchanges import create_exchange
from config.settings import TradingConfig
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()


def create_config() -> TradingConfig:
    """Create trading configuration from .env and defaults"""
    try:
        # Get FORWARD_TESTING from .env (still in .env as requested)
        forward_testing = os.getenv(
            "FORWARD_TESTING", "false").lower() in ("true", "1", "yes")
        # Create config with all defaults from settings.py, overriding FORWARD_TESTING from .env
        config = TradingConfig(FORWARD_TESTING=forward_testing)
        # Validate settings
        if config.LEVERAGE < 1 or config.LEVERAGE > 125:
            raise ValueError("Leverage must be between 1 and 125")
        if config.MARGIN_MODE.upper() not in ['ISOLATED', 'CROSS']:
            raise ValueError("Margin mode must be ISOLATED or CROSS")
        if config.EXCHANGE.upper() not in ['BINANCE', 'BITUNIX']:
            raise ValueError("Exchange must be BINANCE or BITUNIX")
        if config.LLM_PROVIDER.lower() not in ['xai', 'groq', 'openai', 'openrouter', 'deepseek', 'mistral']:
            raise ValueError(
                f"Unsupported LLM provider: {config.LLM_PROVIDER}")
        return config
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        config = create_config()
        # Show configuration
        mode = "PAPER" if config.FORWARD_TESTING else "LIVE"
        print(
            f"Config: {config.EXCHANGE} | {mode} | {config.LLM_PROVIDER} ({config.LLM_MODEL})")
        api_key = os.getenv("EXCHANGE_API_KEY")
        api_secret = os.getenv("EXCHANGE_API_SECRET")
        if not config.FORWARD_TESTING and (not api_key or not api_secret):
            raise ValueError("API keys required for live trading")
        exchange = create_exchange(config, api_key, api_secret)
        # Get initial balance for live trading
        if not config.FORWARD_TESTING:
            try:
                live_balance = exchange.get_account_balance(config.CURRENCY)
                config.INITIAL_CAPITAL = live_balance
                print(f"Live balance: ${live_balance:,.2f} {config.CURRENCY}")
            except Exception as e:
                print(f"Warning: Could not fetch balance: {e}")
        bot = TradingBot(config=config, exchange=exchange)
        bot.run_cycle()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
