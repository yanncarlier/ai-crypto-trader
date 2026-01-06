# run.py
import asyncio
import logging
from core.trader import TradingBot
from exchanges.bitunix import BitunixFutures
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import signal

from typing import Dict, Any
from utils.logger import configure_logger

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



def validate_config(config: Dict[str, Any]) -> bool:
    """Comprehensive validation of configuration parameters"""
    validation_errors = []
    
    try:
        # Trading Configuration Validation
        if not config.get('SYMBOL') or len(config['SYMBOL']) < 4:
            validation_errors.append("SYMBOL must be at least 4 characters")
        
        if config['LEVERAGE'] < 1 or config['LEVERAGE'] > 125:
            validation_errors.append("LEVERAGE must be between 1 and 125")
            
        if config['MARGIN_MODE'].upper() not in ['ISOLATED', 'CROSS']:
            validation_errors.append("MARGIN_MODE must be ISOLATED or CROSS")
            
        if config['CYCLE_MINUTES'] < 0.1 or config['CYCLE_MINUTES'] > 1440:  # 0.1 min to 24 hours
            validation_errors.append("CYCLE_MINUTES must be between 0.1 and 1440")
            
        # Risk Management Validation
        sl_percent = config.get('STOP_LOSS_PERCENT', 0)
        tp_percent = config.get('TAKE_PROFIT_PERCENT', 0)
        
        if sl_percent <= 0 or sl_percent > 50:  # Max 50% stop loss
            validation_errors.append("STOP_LOSS_PERCENT must be > 0 and ≤ 50%")
            
        if tp_percent <= 0 or tp_percent > 100:  # Max 100% take profit
            validation_errors.append("TAKE_PROFIT_PERCENT must be > 0 and ≤ 100%")
            
        if tp_percent < sl_percent * 1.5:  # Minimum 1.5:1 risk-reward
            validation_errors.append("TAKE_PROFIT_PERCENT should be at least 1.5x STOP_LOSS_PERCENT")
            
        # Financial Limits
        max_risk_percent = config.get('MAX_RISK_PERCENT', 0)
        if max_risk_percent <= 0 or max_risk_percent > 10:  # Max 10% risk per trade
            validation_errors.append("MAX_RISK_PERCENT must be > 0 and ≤ 10%")
            
        min_confidence = config.get('MIN_CONFIDENCE', 0)
        if min_confidence < 0.1 or min_confidence > 1.0:
            validation_errors.append("MIN_CONFIDENCE must be between 0.1 and 1.0")
            
        # Exchange Configuration
        if not config.get('FORWARD_TESTING', False):  # Only validate API keys for live trading
            if not config.get('EXCHANGE_API_KEY') or len(config['EXCHANGE_API_KEY']) < 10:
                validation_errors.append("EXCHANGE_API_KEY required for live trading")
                
            if not config.get('EXCHANGE_API_SECRET') or len(config['EXCHANGE_API_SECRET']) < 10:
                validation_errors.append("EXCHANGE_API_SECRET required for live trading")
        
        # AI Configuration
        llm_provider = config.get('LLM_PROVIDER', '').lower()
        valid_providers = ['deepseek', 'xai', 'groq', 'openai', 'openrouter', 'mistral']
        if llm_provider not in valid_providers:
            validation_errors.append(f"LLM_PROVIDER must be one of {valid_providers}")
            
        if config.get('LLM_TEMPERATURE', 0) < 0 or config.get('LLM_TEMPERATURE', 0) > 2:
            validation_errors.append("LLM_TEMPERATURE must be between 0 and 2")
            
        # Technical Indicators
        if config.get('ATR_PERIOD', 0) < 5 or config.get('ATR_PERIOD', 0) > 100:
            validation_errors.append("ATR_PERIOD must be between 5 and 100")
            
        if config.get('RSI_PERIOD', 0) < 5 or config.get('RSI_PERIOD', 0) > 50:
            validation_errors.append("RSI_PERIOD must be between 5 and 50")
            
        if config.get('EMA_PERIOD', 0) < 5 or config.get('EMA_PERIOD', 0) > 200:
            validation_errors.append("EMA_PERIOD must be between 5 and 200")
            
        if config.get('WEEKLY_GROWTH_TARGET', 0) < 0.1 or config.get('WEEKLY_GROWTH_TARGET', 0) > 100:
            validation_errors.append("WEEKLY_GROWTH_TARGET must be between 0.1% and 100%")
            
        # Fees
        if config.get('TAKER_FEE', 0) < 0 or config.get('TAKER_FEE', 0) > 0.01:  # Max 1% fee
            validation_errors.append("TAKER_FEE must be between 0 and 0.01 (1%)")
        
        if validation_errors:
            logging.error("Configuration validation failed:")
            for error in validation_errors:
                logging.error(f"  - {error}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Unexpected error during config validation: {e}")
        return False

def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables with comprehensive validation"""
    try:
        config = {
            # Trading Configuration
            'CRYPTO': get_env_str('CRYPTO', 'BTC'),
            'SYMBOL': get_env_str('SYMBOL', 'BTCUSDT'),
            'CURRENCY': get_env_str('CURRENCY', 'USDT'),
            'CYCLE_MINUTES': get_env_float('CYCLE_MINUTES', 0),
            'LEVERAGE': get_env_int('LEVERAGE', 2),
            'MARGIN_MODE': get_env_str('MARGIN_MODE', 'PLACEHOLDER'),
            'TAKER_FEE': get_env_float('TAKER_FEE', 0),

            # Risk Management
            'VOLATILITY_ADJUSTED': get_env_bool('VOLATILITY_ADJUSTED', False),
            'ATR_PERIOD': get_env_int('ATR_PERIOD', 14),
            'STOP_LOSS_PERCENT': get_env_float('STOP_LOSS_PERCENT', 2.0),
            'TAKE_PROFIT_PERCENT': get_env_float('TAKE_PROFIT_PERCENT', 4.0),

            # AI Prompt Configuration
            'MAX_RISK_PERCENT': get_env_float('MAX_RISK_PERCENT', 1.0),
            'MIN_RISK_REWARD_RATIO': get_env_float('MIN_RISK_REWARD_RATIO', 2.0),
            'CONFIDENCE_THRESHOLD': get_env_float('CONFIDENCE_THRESHOLD', 0.7),
            'WEEKLY_GROWTH_TARGET': get_env_float('WEEKLY_GROWTH_TARGET', 5.0),
            'RSI_PERIOD': get_env_int('RSI_PERIOD', 14),
            'EMA_PERIOD': get_env_int('EMA_PERIOD', 20),
            'BB_PERIOD': get_env_int('BB_PERIOD', 20),
            'LONG_TF_MULTIPLIER': get_env_int('LONG_TF_MULTIPLIER', 4),
            'OHLCV_LIMIT': get_env_int('OHLCV_LIMIT', 7),

            # Configuration
            'FORWARD_TESTING': get_env_bool('FORWARD_TESTING', False),
            'LLM_PROVIDER': get_env_str('LLM_PROVIDER', 'PLACEHOLDER'),
            'LLM_MODEL': get_env_str('LLM_MODEL', 'PLACEHOLDER'),
            'LLM_TEMPERATURE': get_env_float('LLM_TEMPERATURE', 0),
            'LLM_MAX_TOKENS': get_env_int('LLM_MAX_TOKENS', 0),
            'EXCHANGE': get_env_str('EXCHANGE', 'PLACEHOLDER'),
            'EXCHANGE_API_KEY': get_env_str('EXCHANGE_API_KEY', 'PLACEHOLDER'),
            'EXCHANGE_API_SECRET': get_env_str('EXCHANGE_API_SECRET', 'PLACEHOLDER'),
            'LLM_API_KEY': get_env_str('LLM_API_KEY', 'PLACEHOLDER'),
            'MIN_CONFIDENCE': get_env_float('MIN_CONFIDENCE', 0),
            
            # Additional risk controls
            'MIN_BALANCE_THRESHOLD': get_env_float('MIN_BALANCE_THRESHOLD', 10.0),
            'BALANCE_DROP_ALERT_PERCENT': get_env_float('BALANCE_DROP_ALERT_PERCENT', 5.0),
        }

        # Add RUN_NAME property
        mode = "paper" if config['FORWARD_TESTING'] else "LIVE"
        config['RUN_NAME'] = f"{mode}_{config['EXCHANGE']}_{config['CRYPTO']}_{config['SYMBOL']}_{config['CYCLE_MINUTES']}min_{config['LEVERAGE']}x"

        # Validate settings with comprehensive checks
        if not validate_config(config):
            raise ValueError("Configuration validation failed")
            
        return config

    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


async def main():
    try:
        config = get_config()
        configure_logger(config['RUN_NAME'])

        # Initialize app logger for startup info
        app_logger = logging.getLogger("app")

        # Show configuration
        mode = "PAPER" if config['FORWARD_TESTING'] else "LIVE"
        print(f"Bitunix Futures Trader | {mode} | {config['LLM_PROVIDER']}")
        print(f"Symbol: {config['SYMBOL']} | Leverage: {config['LEVERAGE']}x")
        print(f"Cycle: {config['CYCLE_MINUTES']:.1f} minutes")

        app_logger.info(f"Bot started | Mode: {mode} | Symbol: {config['SYMBOL']} | Leverage: {config['LEVERAGE']}x | Cycle: {config['CYCLE_MINUTES']:.1f}min")

        if config['FORWARD_TESTING']:
            print("Running in FORWARD TESTING mode (real data, no execution - notifications only)")
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

            exchange = BitunixFutures(api_key, api_secret, config)

            try:
                # Start position monitoring
                await exchange.start_monitoring()

                # Get account summary
                account_summary = await exchange.get_account_summary(config['CURRENCY'], config['SYMBOL'])
                live_balance = account_summary['balance']
                equity = account_summary['equity']

                if live_balance <= 0:
                    print(f"❌ Error: Account balance is ${live_balance:,.2f}. Cannot start trading with zero or negative balance.")
                    print("Please check your Bitunix API credentials and ensure your account has sufficient funds.")
                    sys.exit(1)

                print(f"Balance: ${live_balance:,.2f} {config['CURRENCY']}")
                print(f"Equity: ${equity:,.2f} {config['CURRENCY']}")

                if 'unrealized_pnl' in account_summary:
                    pnl = account_summary['unrealized_pnl']
                    if pnl != 0:
                        pnl_pct = (pnl / live_balance) * 100
                        pnl_sign = '+' if pnl > 0 else ''
                        print(
                            f"   • Unrealized PnL: {pnl_sign}${abs(pnl):,.2f} ({pnl_sign}{abs(pnl_pct):.2f}%)")

                # Check for existing positions and close them on startup
                position = await exchange.get_pending_positions(config['SYMBOL'])
                if position:
                    print(f"Found existing position: {position.side} {position.size} @ ${position.entry_price:,.0f}")
                    await exchange.close_position(position, "Startup cleanup")
                    print("Closed existing position on startup")
            except Exception as e:
                print(f"Could not fetch balance: {e}")
                raise

        # Initialize and run the trading bot
        bot = TradingBot(config=config, exchange=exchange)

        while True:
            await bot.run_cycle()
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
            loop.run_until_complete(asyncio.gather(
                *pending, return_exceptions=True))

        # Close the loop
        loop.close()
        print("Bot stopped")
