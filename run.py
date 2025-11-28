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
# Toggle paper/live here
# ← SET TO False WHEN READY FOR LIVE
config = TradingConfig(FORWARD_TESTING=True)
# Choose exchange
if config.FORWARD_TESTING:
    exchange = ForwardTester(config)
else:
    exchange = BitunixFutures(
        api_key=os.getenv("EXCHANGE_API_KEY"),
        api_secret=os.getenv("EXCHANGE_API_SECRET")
    )
# Create bot — no more llm_key argument!
bot = TradingBot(config=config, exchange=exchange)
# Run one cycle (or put in a loop / scheduler later)
if __name__ == "__main__":
    bot.run_cycle()
