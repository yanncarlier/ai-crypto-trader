from dotenv import load_dotenv
import os
from config.settings import TradingConfig
from exchanges.forward_tester import ForwardTester
from exchanges.bitunix import BitunixFutures
from core.trader import TradingBot
load_dotenv()
# Toggle paper/live here
config = TradingConfig(FORWARD_TESTING=True)  # ← SET TO False FOR LIVE!
exchange = ForwardTester(config) if config.FORWARD_TESTING else BitunixFutures(
    os.getenv("EXCHANGE_API_KEY"),
    os.getenv("EXCHANGE_API_SECRET")
)
bot = TradingBot(
    config=config,
    exchange=exchange,
    llm_key=os.getenv("LLM_API_KEY")
)
if __name__ == "__main__":
    bot.run_cycle()
