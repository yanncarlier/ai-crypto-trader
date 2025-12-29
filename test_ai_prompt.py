#!/usr/bin/env python3
"""
AI Prompt Test Script

This script tests the AI prompt building and API response functionality.
Run this to verify your AI API credentials and prompt generation are working.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from ai.prompt_builder import build_prompt
from ai.provider import send_request

load_dotenv()


async def test_ai_prompt():
    """Test AI prompt building and API response."""

    print("🤖 Testing AI Prompt and API Response")
    print("=" * 50)

    # Load configuration
    api_key = os.getenv('LLM_API_KEY')
    provider = os.getenv('LLM_PROVIDER', 'deepseek')
    model = os.getenv('LLM_MODEL', 'default')
    symbol = os.getenv('SYMBOL', 'BTCUSDT')
    currency = os.getenv('CURRENCY', 'USDT')

    if not api_key:
        print("❌ ERROR: Missing LLM_API_KEY in .env file")
        print("   Make sure LLM_API_KEY is set")
        return False

    print(f"📋 Configuration:")
    print(f"   Provider: {provider}")
    print(f"   Model: {model}")
    print(f"   Symbol: {symbol}")
    print(f"   Currency: {currency}")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
    print()

    try:
        # Create sample data for prompt building
        print("📝 Building sample prompt...")

        # Sample configuration
        config = {
            'SYMBOL': symbol,
            'CURRENCY': currency,
            'CRYPTO': 'Bitcoin',
            'LEVERAGE': 20,
            'TAKER_FEE': 0.0006,
            'CYCLE_MINUTES': 5,
            'MAX_POSITION_SIZE_PCT': 0.20,
            'DAILY_LOSS_LIMIT_PCT': 0.20,
            'MAX_DRAWDOWN_PCT': 0.15,
            'MAX_HOLD_HOURS': 10000,
            'LLM_PROVIDER': provider,
            'LLM_MODEL': model,
            'LLM_TEMPERATURE': 0.3,
            'LLM_MAX_TOKENS': 2000,
            'LLM_API_KEY': api_key
        }

        # Sample timestamp and elapsed time
        timestamp = datetime.now()
        minutes_elapsed = 5

        # Sample account data
        account_balance = 1000.0
        equity = 1000.0

        # Sample open positions
        open_positions = [
            {
                'side': 'LONG',
                'size': 0.001,
                'entry_price': 45000.0,
                'unrealized_pnl': 25.0
            }
        ]

        # Sample price history (OHLCV format)
        price_history_short = [
            {'timestamp': '2024-01-01 12:00', 'open': 44000, 'high': 44500, 'low': 43900, 'close': 44200, 'volume': 100},
            {'timestamp': '2024-01-01 12:05', 'open': 44200, 'high': 44800, 'low': 44100, 'close': 44600, 'volume': 120},
            {'timestamp': '2024-01-01 12:10', 'open': 44600, 'high': 44900, 'low': 44400, 'close': 44700, 'volume': 110},
            {'timestamp': '2024-01-01 12:15', 'open': 44700, 'high': 45000, 'low': 44600, 'close': 44850, 'volume': 130},
            {'timestamp': '2024-01-01 12:20', 'open': 44850, 'high': 45200, 'low': 44700, 'close': 45000, 'volume': 140},
        ]

        price_history_long = [
            {'timestamp': '2024-01-01 11:00', 'open': 43000, 'high': 44500, 'low': 42800, 'close': 44200, 'volume': 500},
            {'timestamp': '2024-01-01 11:20', 'open': 44200, 'high': 44800, 'low': 44000, 'close': 44600, 'volume': 520},
            {'timestamp': '2024-01-01 11:40', 'open': 44600, 'high': 44900, 'low': 44400, 'close': 44700, 'volume': 510},
            {'timestamp': '2024-01-01 12:00', 'open': 44700, 'high': 45000, 'low': 44600, 'close': 44850, 'volume': 530},
        ]

        # Sample indicators
        indicators = {
            'price': 45000.0,
            'live_price': 45050.0,
            'price_diff': 50.0,
            'price_diff_pct': 0.11,
            'RSI': 65.5,
            'MACD': {'hist': 25.3, 'signal': 18.7},
            'EMA_20': 44750.0,
            'BB_upper': 45200.0,
            'BB_middle': 44800.0,
            'BB_lower': 44400.0
        }

        # Sample predictive signals
        predictive_signals = {
            'volatility': 'Medium',
            'order_book_depth_bid': 1500000,
            'order_book_depth_ask': 1450000,
            'sentiment_proxy': 'Bullish'
        }

        # Build the prompt
        prompt = build_prompt(
            timestamp=timestamp,
            minutes_elapsed=minutes_elapsed,
            account_balance=account_balance,
            equity=equity,
            open_positions=open_positions,
            price_history_short=price_history_short,
            price_history_long=price_history_long,
            indicators=indicators,
            predictive_signals=predictive_signals,
            config=config
        )

        print("✅ Prompt built successfully")
        print(f"   Prompt length: {len(prompt)} characters")
        print()

        # Test AI API request
        print("🔗 Sending request to AI provider...")
        try:
            outlook = await send_request(prompt, config)
            print("✅ AI response received successfully")
            print()

            # Display results
            print("📊 AI Analysis Results:")
            print(f"   Interpretation: {outlook.interpretation}")
            print(f"   Confidence: {outlook.confidence:.2f}")
            if outlook.action:
                print(f"   Action: {outlook.action}")
            print(f"   Reasons: {outlook.reasons}")
            print()

            # Success
            print("🎉 AI PROMPT TEST PASSED!")
            print("✅ Your AI API credentials are working correctly")
            print("✅ Prompt generation is functioning properly")
            print()
            print("💡 Next steps:")
            print("   1. Review the AI response above")
            print("   2. Run the full trading bot: python run.py")
            print("   3. Monitor AI decisions in the logs")

            return True

        except Exception as e:
            print(f"❌ Failed to get AI response: {e}")
            print("   This could be due to:")
            print("   - Invalid API key")
            print("   - Network connectivity issues")
            print("   - API rate limits")
            print("   - Invalid provider/model configuration")
            return False

    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    success = await test_ai_prompt()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    print("AI Prompt Test")
    print("This will test your AI prompt generation and API connection")
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
