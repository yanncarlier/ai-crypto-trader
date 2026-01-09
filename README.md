## Heads up: The project is currently unstable/broken and on hiatus.

## I'm not planning to work on it in the foreseeable future—proceed with caution!

## Contributions and feedback are welcome and greatly appreciated!! if I have time I will work on it.

Use at Your Own Risk
This is experimental/beta software. It may contain bugs or cause unexpected behavior.
No warranties are provided. Use entirely at your own discretion and risk.  

Bitunix API Documentation  
https://openapidoc.bitunix.com/doc/common/introduction.html  

Official OpenAPI SDK  
https://github.com/BitunixOfficial/open-api  

# AI Crypto Trader

Modular AI-powered BTC/USDT futures bot with paper & live mode.

### Features
- Grok / GPT / Claude compatible
- Paper trading with realistic simulation
- Clean separation of concerns
- Easy to extend (Binance, Bybit, etc.)

### Setup
```bash
uv run main.py
source .venv/bin/activate
uv pip install -r requirements.txt
cp example.env .env
# 1. Add your keys
.venv/bin/python ./run.py
```



```bash
# 2. When you’re ready for live (double-check keys first!)
#    → edit settings.py line 15:
#    FORWARD_TESTING: bool = False

# 3. Optional: run every 10 minutes automatically
#    (add this to crontab -e)
*/10 * * * * cd ai-crypto-trader/ && bash batch_runner.sh >> logs/cron.log 2>&1
```
