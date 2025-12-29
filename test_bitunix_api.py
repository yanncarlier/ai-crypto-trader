#!/usr/bin/env python3
"""
Bitunix API Credentials Test Script

This script tests your Bitunix API credentials and basic functionality.
Run this before starting the trading bot to ensure your API keys are working.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from exchanges.bitunix import BitunixFutures

load_dotenv()


async def test_bitunix_api():
    """Test Bitunix API credentials and basic functionality."""

    print("🔍 Testing Bitunix API Credentials")
    print("=" * 50)

    # Load configuration
    api_key = os.getenv('EXCHANGE_API_KEY')
    api_secret = os.getenv('EXCHANGE_API_SECRET')
    symbol = os.getenv('SYMBOL', 'BTCUSDT')
    currency = os.getenv('CURRENCY', 'USDT')

    if not api_key or not api_secret:
        print("❌ ERROR: Missing API credentials in .env file")
        print("   Make sure EXCHANGE_API_KEY and EXCHANGE_API_SECRET are set")
        return False

    print(f"📋 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Currency: {currency}")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
    print()

    try:
        # Create exchange instance
        print("🔗 Connecting to Bitunix...")
        config = {
            'SYMBOL': symbol,
            'CURRENCY': currency,
            'LEVERAGE': 20,
            'MARGIN_MODE': 'ISOLATED',
            'MAX_POSITION_SIZE_PCT': 0.20,  # 20%
            'DAILY_LOSS_LIMIT_PCT': 0.20,  # 20%
            'MAX_DRAWDOWN_PCT': 0.15,      # 15%
            'MAX_HOLD_HOURS': 10000,       # Long time
            'VOLATILITY_ADJUSTED': True,
            'ATR_PERIOD': 14,
            'MIN_LIQUIDITY': 1000000
        }
        exchange = BitunixFutures(api_key, api_secret, config)
        print("✅ Exchange instance created successfully")
        print()

        # Test 1: Get current price
        print("📈 Testing price fetch...")
        try:
            price = await exchange.get_current_price(symbol)
            print(f"✅ Current {symbol} price: ${price:,.2f}")
        except Exception as e:
            print(f"❌ Failed to get price: {e}")
            return False
        print()

        # Test 2: Get account balance
        print("💰 Testing account balance...")
        try:
            balance = await exchange.get_account_balance(currency)
            print(f"✅ Account balance: ${balance:,.2f} {currency}")

            if balance <= 0:
                print("⚠️  WARNING: Account balance is zero or negative")
                print("   Make sure your Bitunix account has sufficient funds")
            else:
                print("🎉 Account has trading funds available!")

        except Exception as e:
            print(f"❌ Failed to get balance: {e}")
            print("   This could be due to:")
            print("   - Invalid API credentials")
            print("   - Network connectivity issues")
            print("   - API permissions not set correctly")
            return False
        print()

        # Test 3: Get account summary
        print("📊 Testing account summary...")
        try:
            summary = await exchange.get_account_summary(currency, symbol)
            print("✅ Account summary retrieved:")
            print(f"   Balance: ${summary['balance']:,.2f}")
            print(f"   Equity: ${summary['equity']:,.2f}")
            print(f"   Positions: {summary['total_positions']}")
        except Exception as e:
            print(f"❌ Failed to get account summary: {e}")
            return False
        print()

        # Test 4: Get market data
        print("📊 Testing market data...")
        try:
            ohlcv = await exchange.get_ohlcv(symbol, timeframe=60, limit=5)
            if ohlcv and len(ohlcv) > 0:
                print(f"✅ Market data retrieved: {len(ohlcv)} candles")
                latest = ohlcv[-1]
                print(f"   Latest candle: O={latest[1]:.2f}, H={latest[2]:.2f}, L={latest[3]:.2f}, C={latest[4]:.2f}")
            else:
                print("❌ No market data received")
                return False
        except Exception as e:
            print(f"❌ Failed to get market data: {e}")
            return False
        print()

        # Test 5: Check positions
        print("📋 Testing position check...")
        try:
            position = await exchange.get_pending_positions(symbol)
            if position:
                print("✅ Open position found:")
                print(f"   Side: {position.side}")
                print(f"   Size: {position.size}")
                print(f"   Entry Price: ${position.entry_price:,.2f}")
            else:
                print("✅ No open positions (ready for trading)")
        except Exception as e:
            print(f"❌ Failed to check positions: {e}")
            return False
        print()

        # Success
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Bitunix API credentials are working correctly")
        print("✅ You can now run the trading bot")
        print()
        print("💡 Next steps:")
        print("   1. Make sure you have sufficient funds in your Bitunix account")
        print("   2. Run the trading bot: python run.py")
        print("   3. Monitor the logs for any issues")

        return True

    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            if 'exchange' in locals():
                await exchange.close()
        except:
            pass


async def main():
    """Main function."""
    success = await test_bitunix_api()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    print("Bitunix API Credentials Test")
    print("This will test your API connection and credentials")
    print()

    # Run the async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    finally:
        loop.close()
