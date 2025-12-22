# run.py
import asyncio
from core.trader import TradingBot
from exchanges.bitunix import BitunixFutures
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import signal
from exchanges.forward_tester import ForwardTester
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    val = os.getenv(name, '').lower()
    if val in ('true', '1', 't', 'y', 'yes'):
        return True
    elif val in ('false', '0', 'f', 'n', 'no', ''):
        return default
    return default

def get_env_float(name: str, default: float) -> float:
    """Get float value from environment variable"""
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default

def get_env_int(name: str, default: int) -> int:
    """Get integer value from environment variable"""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default

def get_env_str(name: str, default: str) -> str:
    """Get string value from environment variable"""
    return os.getenv(name, default)

def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    try:
        config = {
            # Trading Configuration
            'CRYPTO': 'Bitcoin',
            'SYMBOL': get_env_str('SYMBOL', 'BTCUSDT'),
            'CURRENCY': get_env_str('CURRENCY', 'USDT'),
            'CYCLE_MINUTES': get_env_int('CYCLE_MINUTES', 10),
            'LEVERAGE': get_env_int('LEVERAGE', 2),
            'MARGIN_MODE': get_env_str('MARGIN_MODE', 'ISOLATED'),
            'POSITION_SIZE': get_env_str('POSITION_SIZE', '10%'),
            'STOP_LOSS_PERCENT': get_env_float('STOP_LOSS_PERCENT', 10.0) or None,
            'TAKE_PROFIT_PERCENT': get_env_float('TAKE_PROFIT_PERCENT', 0) or None,
            'INITIAL_CAPITAL': get_env_float('INITIAL_CAPITAL', 10000.0),
            'TAKER_FEE': get_env_float('TAKER_FEE', 0.0006),
            
            # Risk Management
            'MAX_POSITION_SIZE_PCT': get_env_float('MAX_POSITION_SIZE_PCT', 10.0) / 100,
            'DAILY_LOSS_LIMIT_PCT': get_env_float('DAILY_LOSS_LIMIT_PCT', 2.0) / 100,
            'MAX_DRAWDOWN_PCT': get_env_float('MAX_DRAWDOWN_PCT', 5.0) / 100,
            'MAX_HOLD_HOURS': get_env_int('MAX_HOLD_HOURS', 24),
            
            # Configuration
            'FORWARD_TESTING': get_env_bool('FORWARD_TESTING', False),
            'LLM_PROVIDER': get_env_str('LLM_PROVIDER', 'deepseek'),
            'LLM_MODEL': get_env_str('LLM_MODEL', 'default'),
            'LLM_TEMPERATURE': get_env_float('LLM_TEMPERATURE', 0.2),
            'LLM_MAX_TOKENS': get_env_int('LLM_MAX_TOKENS', 800),
            'EXCHANGE': get_env_str('EXCHANGE', 'BITUNIX'),
            'TEST_NET': get_env_bool('TEST_NET', False),
            'EXCHANGE_API_KEY': get_env_str('EXCHANGE_API_KEY', ''),
            'EXCHANGE_API_SECRET': get_env_str('EXCHANGE_API_SECRET', ''),
            'LLM_API_KEY': get_env_str('LLM_API_KEY', ''),
        }
        
        # Add RUN_NAME property
        mode = "paper" if config['FORWARD_TESTING'] else "LIVE"
        config['RUN_NAME'] = f"{mode}_{config['EXCHANGE']}_{config['CRYPTO']}_{config['SYMBOL']}_{config['CYCLE_MINUTES']}min_{config['LEVERAGE']}x"
        
        # Validate settings
        if config['LEVERAGE'] < 1 or config['LEVERAGE'] > 125:
            raise ValueError("Leverage must be between 1 and 125")
        if config['MARGIN_MODE'].upper() not in ['ISOLATED', 'CROSS']:
            raise ValueError("Margin mode must be ISOLATED or CROSS")
        if config['CYCLE_MINUTES'] < 1:
            raise ValueError("CYCLE_MINUTES must be at least 1")
            
        return config
        
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

async def main():
    try:
        config = get_config()
        # Show configuration
        mode = "PAPER" if config['FORWARD_TESTING'] else "LIVE"
        print(f"Bitunix Futures Trader | {mode} | {config['LLM_PROVIDER']}")
        print(f"Symbol: {config['SYMBOL']} | Leverage: {config['LEVERAGE']}x")
        print(f"Position: {config['POSITION_SIZE']} | Stop Loss: {config['STOP_LOSS_PERCENT']}%")
        print(f"Cycle: {config['CYCLE_MINUTES']} minutes")
        
        if config['FORWARD_TESTING']:
            print("Running in PAPER TRADING mode (using live exchange dummy)")
            api_key = config['EXCHANGE_API_KEY']
            api_secret = config['EXCHANGE_API_SECRET']
            exchange = BitunixFutures(api_key, api_secret, config)
        else:
            # Live trading with Bitunix
            api_key = config['EXCHANGE_API_KEY']
            api_secret = config['EXCHANGE_API_SECRET']
            if not api_key or not api_secret:
                raise ValueError("Bitunix API keys required for live trading")
                
            print("LIVE TRADING with Bitunix Futures")
            print("Risk Management Settings:")
            print(f"   • Max Position Size: {config['MAX_POSITION_SIZE_PCT']*100:.1f}%")
            print(f"   • Daily Loss Limit: {config['DAILY_LOSS_LIMIT_PCT']*100:.1f}%")
            print(f"   • Max Drawdown: {config['MAX_DRAWDOWN_PCT']*100:.1f}%")
            print(f"   • Max Hold Time: {config['MAX_HOLD_HOURS']} hours")
            
            exchange = BitunixFutures(api_key, api_secret, config)
            
            try:
                # Start position monitoring
                await exchange.start_monitoring()
                
                # Get account summary
                account_summary = await exchange.get_account_summary(config['CURRENCY'], config['SYMBOL'])
                live_balance = account_summary['balance']
                equity = account_summary['equity']
                config['INITIAL_CAPITAL'] = live_balance
                
                print(f"Balance: ${live_balance:,.2f} {config['CURRENCY']}")
                print(f"Equity: ${equity:,.2f} {config['CURRENCY']}")
                
                if 'unrealized_pnl' in account_summary:
                    pnl = account_summary['unrealized_pnl']
                    if pnl != 0:
                        pnl_pct = (pnl / live_balance) * 100
                        pnl_sign = '+' if pnl > 0 else ''
                        print(f"   • Unrealized PnL: {pnl_sign}${abs(pnl):,.2f} ({pnl_sign}{abs(pnl_pct):.2f}%)")
            except Exception as e:
                print(f"Could not fetch balance: {e}")
                raise

        # Initialize and run the trading bot
        bot = TradingBot(config=config, exchange=exchange)
        
        while True:
            await bot.run_cycle()
            print(f"--- Cycle finished, waiting {config['CYCLE_MINUTES']} minutes ---")
            await asyncio.sleep(config['CYCLE_MINUTES'] * 60)

    except Exception as e:
        print(f"Error: {e}")
        if 'exchange' in locals():
            try:
                if hasattr(exchange, 'close'):
                    await exchange.close()
                elif hasattr(exchange, 'close_connection'):
                    await exchange.close_connection()
            except Exception as close_error:
                print(f"Error during cleanup: {close_error}")
        sys.exit(1)

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    # Additional cleanup if needed
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async main function
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up any running tasks
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        
        # Run the loop until all tasks are cancelled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        # Close the loop
        loop.close()
        print("Bot stopped")
