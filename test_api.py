#!/usr/bin/env python3
"""
Bitunix API Test Script

This script tests the connection to the Bitunix exchange API using your API keys.
It verifies both public and private endpoints if API keys are provided.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("api_test")

def test_connection():
    """Test connection to Bitunix API"""
    try:
        from exchanges.bitunix import BitunixFutures
        
        logger.info("Initializing Bitunix client...")
        exchange = BitunixFutures(
            api_key=os.getenv("EXCHANGE_API_KEY"),
            api_secret=os.getenv("EXCHANGE_API_SECRET")
        )
        
        # Test public endpoints
        logger.info("Testing public endpoints...")
        ticker = exchange.get_ticker("BTCUSDT")
        logger.info(f"✅ Public API test successful")
        logger.info(f"📈 BTC/USDT Ticker: {ticker}")
        
        # Test private endpoints if API keys are provided
        if os.getenv("EXCHANGE_API_KEY") and os.getenv("EXCHANGE_API_SECRET"):
            logger.info("Testing private endpoints...")
            try:
                balance = exchange.get_account_balance("USDT")
                logger.info(f"✅ Private API test successful")
                logger.info(f"💰 USDT Balance: {balance}")
                
                # Test fetching open positions
                positions = exchange.get_pending_positions("BTCUSDT")
                logger.info(f"📊 Open positions: {positions if positions else 'None'}")
                
            except Exception as e:
                logger.error(f"❌ Private API test failed: {str(e)}")
                logger.warning("This could be due to insufficient permissions or invalid API keys")
                
    except ImportError as e:
        logger.error(f"Failed to import Bitunix module: {e}")
        logger.info("Make sure you're running from the project root directory")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check if running in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.warning("⚠️  Not running in a virtual environment. It's recommended to use one.")
    
    # Check required environment variables
    if not os.getenv("EXCHANGE_API_KEY") or not os.getenv("EXCHANGE_API_SECRET"):
        logger.warning("⚠️  EXCHANGE_API_KEY and/or EXCHANGE_API_SECRET not found in .env")
        logger.info("Testing only public endpoints...")
    
    # Run the test
    success = test_connection()
    
    if success:
        logger.info("✅ All tests completed successfully!")
    else:
        logger.error("❌ Some tests failed. Please check the logs above for details.")
        sys.exit(1)
