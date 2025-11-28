# run.py
from core.trader import TradingBot
from exchanges.bitunix import BitunixFutures
from exchanges.forward_tester import ForwardTester
from config.settings import TradingConfig
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()
# Toggle paper/live here → SET TO False WHEN READY FOR LIVE
FORWARD_TESTING = os.getenv(
    "FORWARD_TESTING", "false").lower() in ("true", "1", "yes")
config = TradingConfig(FORWARD_TESTING=FORWARD_TESTING)
if config.FORWARD_TESTING:
    exchange = ForwardTester(config)
else:
    api_key = os.getenv("EXCHANGE_API_KEY")
    api_secret = os.getenv("EXCHANGE_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError(
            "EXCHANGE_API_KEY and EXCHANGE_API_SECRET required for live trading")
    exchange = BitunixFutures(api_key=api_key, api_secret=api_secret)
bot = TradingBot(config=config, exchange=exchange)
if __name__ == "__main__":
    bot.run_cycle()
