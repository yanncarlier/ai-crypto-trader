# run.py
from core.trader import TradingBot
from exchanges.bitunix import BitunixFutures
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
        # Get FORWARD_TESTING from .env
        forward_testing = os.getenv(
            "FORWARD_TESTING", "false").lower() in ("true", "1", "yes")
        # Create config with all defaults from settings.py
        config = TradingConfig(
            FORWARD_TESTING=forward_testing,
            EXCHANGE="BITUNIX",  # Force Bitunix
            TEST_NET=False,  # Bitunix doesn't have testnet
        )
        # Validate settings
        if config.LEVERAGE < 1 or config.LEVERAGE > 125:
            raise ValueError("Leverage must be between 1 and 125")
        if config.MARGIN_MODE.upper() not in ['ISOLATED', 'CROSS']:
            raise ValueError("Margin mode must be ISOLATED or CROSS")
        if config.CYCLE_MINUTES < 1:
            raise ValueError("CYCLE_MINUTES must be at least 1")
        return config
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        config = create_config()
        # Show configuration
        mode = "PAPER" if config.FORWARD_TESTING else "LIVE"
        print(f"⚡ Bitunix Futures Trader | {mode} | {config.LLM_PROVIDER}")
        print(f"📊 Symbol: {config.SYMBOL} | Leverage: {config.LEVERAGE}x")
        print(
            f"📈 Position: {config.POSITION_SIZE} | Stop Loss: {config.STOP_LOSS_PERCENT}%")
        print(f"⏱️  Cycle: {config.CYCLE_MINUTES} minutes")
        if config.FORWARD_TESTING:
            print("📝 Running in PAPER TRADING mode")
            from exchanges.forward_tester import ForwardTester
            exchange = ForwardTester(config)
        else:
            # Live trading with Bitunix
            api_key = os.getenv("EXCHANGE_API_KEY")
            api_secret = os.getenv("EXCHANGE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError("Bitunix API keys required for live trading")
            print("🚀 LIVE TRADING with Bitunix Futures")
            exchange = BitunixFutures(api_key, api_secret)
            # Get initial balance
            try:
                live_balance = exchange.get_account_balance(config.CURRENCY)
                config.INITIAL_CAPITAL = live_balance
                print(f"💰 Balance: ${live_balance:,.2f} {config.CURRENCY}")
            except Exception as e:
                print(f"⚠️ Could not fetch balance: {e}")
        bot = TradingBot(config=config, exchange=exchange)
        bot.run_cycle()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
