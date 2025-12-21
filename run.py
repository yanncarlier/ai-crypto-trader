# run.py
import asyncio
from core.trader import TradingBot
from exchanges.bitunix import BitunixFutures
from config.settings import TradingConfig
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import signal

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

def create_config() -> TradingConfig:
    """Create trading configuration from .env and defaults"""
    try:
        # Get FORWARD_TESTING from .env
        forward_testing = os.getenv(
            "FORWARD_TESTING", "false").lower() in ("true", "1", "yes")
        
        # Risk management parameters
        max_position_size_pct = float(os.getenv("MAX_POSITION_SIZE_PCT", "10")) / 100
        daily_loss_limit_pct = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "2")) / 100
        max_drawdown_pct = float(os.getenv("MAX_DRAWDOWN_PCT", "5")) / 100
        max_hold_hours = int(os.getenv("MAX_HOLD_HOURS", "24"))
        
        # Create config with all defaults from settings.py
        config = TradingConfig(
            FORWARD_TESTING=forward_testing,
            EXCHANGE="BITUNIX",  # Force Bitunix
            TEST_NET=False,  # Bitunix doesn't have testnet
            # Risk management settings
            MAX_POSITION_SIZE_PCT=max_position_size_pct,
            DAILY_LOSS_LIMIT_PCT=daily_loss_limit_pct,
            MAX_DRAWDOWN_PCT=max_drawdown_pct,
            MAX_HOLD_HOURS=max_hold_hours,
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

async def main():
    try:
        config = create_config()
        # Show configuration
        mode = "PAPER" if config.FORWARD_TESTING else "LIVE"
        print(f"⚡ Bitunix Futures Trader | {mode} | {config.LLM_PROVIDER}")
        print(f"📊 Symbol: {config.SYMBOL} | Leverage: {config.LEVERAGE}x")
        print(f"📈 Position: {config.POSITION_SIZE} | Stop Loss: {config.STOP_LOSS_PERCENT}%")
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
            print("⚡ Risk Management Settings:")
            print(f"   • Max Position Size: {config.MAX_POSITION_SIZE_PCT*100:.1f}%")
            print(f"   • Daily Loss Limit: {config.DAILY_LOSS_LIMIT_PCT*100:.1f}%")
            print(f"   • Max Drawdown: {config.MAX_DRAWDOWN_PCT*100:.1f}%")
            print(f"   • Max Hold Time: {config.MAX_HOLD_HOURS} hours")
            
            exchange = BitunixFutures(api_key, api_secret, config)
            
            try:
                # Start position monitoring
                await exchange.start_monitoring()
                
                # Get account summary
                account_summary = exchange.get_account_summary(config.CURRENCY, config.SYMBOL)
                live_balance = account_summary['balance']
                equity = account_summary['equity']
                config.INITIAL_CAPITAL = live_balance
                
                print(f"💰 Balance: ${live_balance:,.2f} {config.CURRENCY}")
                print(f"📊 Equity: ${equity:,.2f} {config.CURRENCY}")
                
                if 'unrealized_pnl' in account_summary:
                    pnl = account_summary['unrealized_pnl']
                    if pnl != 0:
                        pnl_pct = (pnl / live_balance) * 100
                        pnl_sign = '+' if pnl > 0 else ''
                        print(f"   • Unrealized PnL: {pnl_sign}${abs(pnl):,.2f} ({pnl_sign}{abs(pnl_pct):.2f}%)")
            except Exception as e:
                print(f"⚠️ Could not fetch balance: {e}")
                raise

        # Initialize and run the trading bot
        bot = TradingBot(config=config, exchange=exchange)
        # Run the trading cycle (synchronous call)
        bot.run_cycle()

    except Exception as e:
        print(f"❌ Error: {e}")
        if 'exchange' in locals():
            try:
                if hasattr(exchange, 'close'):
                    await exchange.close()
                elif hasattr(exchange, 'close_connection'):
                    await exchange.close_connection()
            except Exception as close_error:
                print(f"⚠️ Error during cleanup: {close_error}")
        sys.exit(1)

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    # Additional cleanup if needed
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Bot stopped")