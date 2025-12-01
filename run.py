# run.py
from core.trader import TradingBot
from exchanges import create_exchange
from config.settings import TradingConfig
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()


def validate_config() -> TradingConfig:
    """Validate and create trading configuration"""
    try:
        config = TradingConfig(
            FORWARD_TESTING=os.getenv(
                "FORWARD_TESTING", "false").lower() in ("true", "1", "yes"),
            EXCHANGE=os.getenv("EXCHANGE", "BINANCE"),
            TEST_NET=os.getenv("TEST_NET", "true").lower() in (
                "true", "1", "yes")
        )
        # Basic validation
        if config.LEVERAGE < 1 or config.LEVERAGE > 125:
            raise ValueError("Leverage must be between 1 and 125")
        if config.MARGIN_MODE.upper() not in ['ISOLATED', 'CROSS']:
            raise ValueError("Margin mode must be ISOLATED or CROSS")
        if config.EXCHANGE.upper() not in ['BINANCE', 'BITUNIX']:
            raise ValueError("Exchange must be BINANCE or BITUNIX")
        return config
    except Exception as e:
        logging.error(f"❌ Configuration validation failed: {e}")
        raise


def get_live_balance(exchange, currency: str) -> float:
    """Fetch actual balance from exchange for live trading"""
    try:
        balance = exchange.get_account_balance(currency)
        logging.info(
            f"💰 Successfully fetched live balance: ${balance:,.2f} {currency}")
        return balance
    except Exception as e:
        logging.error(f"❌ Failed to fetch live balance: {e}")
        logging.warning("🔄 Using configured initial capital as fallback")
        return None


if __name__ == "__main__":
    try:
        config = validate_config()
        # Create exchange first to get initial logging
        api_key = os.getenv("EXCHANGE_API_KEY")
        api_secret = os.getenv("EXCHANGE_API_SECRET")
        if not config.FORWARD_TESTING and (not api_key or not api_secret):
            raise ValueError(
                "EXCHANGE_API_KEY and EXCHANGE_API_SECRET required for live trading")
        exchange = create_exchange(config, api_key, api_secret)
        # Now configure logger with the run name
        from utils.logger import configure_logger
        configure_logger(config.RUN_NAME)
        # Display startup information
        if config.FORWARD_TESTING:
            logging.info("🎯 Starting in PAPER TRADING mode")
            logging.info(
                f"💰 Paper Trading Capital: ${config.INITIAL_CAPITAL:,.2f}")
        else:
            logging.info("🎯 Starting in LIVE TRADING mode")
            logging.info(f"🔗 Exchange: {config.EXCHANGE}")
            if config.EXCHANGE == "BINANCE":
                logging.info(f"🔧 Testnet: {config.TEST_NET}")
            # Fetch and display actual balance for live trading
            actual_balance = get_live_balance(exchange, config.CURRENCY)
            if actual_balance is not None:
                # Update the config with the actual balance for position sizing
                config.INITIAL_CAPITAL = actual_balance
                logging.info(
                    f"💰 LIVE ACCOUNT BALANCE: ${actual_balance:,.2f} {config.CURRENCY}")
            else:
                logging.warning(
                    f"⚠️ Using fallback capital: ${config.INITIAL_CAPITAL:,.2f}")
        logging.info(f"⚙️ Trading Pair: {config.SYMBOL}")
        logging.info(f"📈 Cycle: {config.CYCLE_MINUTES} minutes")
        logging.info(f"⚡ Leverage: {config.LEVERAGE}x")
        logging.info(f"🛡️ Margin Mode: {config.MARGIN_MODE}")
        logging.info(f"📊 Position Size: {config.POSITION_SIZE}")
        logging.info(f"🚨 Stop Loss: {config.STOP_LOSS_PERCENT}%")
        bot = TradingBot(config=config, exchange=exchange)
        bot.run_cycle()
    except Exception as e:
        # Basic logging even if logger isn't configured
        print(f"❌ Failed to start trading bot: {e}")
        sys.exit(1)
