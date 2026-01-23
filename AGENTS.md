# AGENTS.md - Development Guidelines for AI Crypto Trader

## Overview
This document provides comprehensive guidelines for AI agents working on the `ai-crypto-trader` codebase. Follow these rules to maintain code quality, consistency, and safety.

## Project Structure
```
ai-crypto-trader/
├── core/           # Core trading logic
├── exchanges/      # Exchange integrations (direct API calls only)
├── ai/            # AI provider and prompt building
├── utils/         # Shared utilities (logging, risk management)
├── tests/         # Unit tests
├── main.py        # Entry point
├── run.py         # Main trading application
└── requirements.txt
```

## Build/Lint/Test Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_prompt_and_provider.py

# Run single test function
python -m pytest tests/test_prompt_and_provider.py::test_build_prompt_minimal_config -v

# Run tests with coverage
python -m pytest tests/ --cov=core --cov=ai --cov=utils --cov-report=html
```

### Running the Application
```bash
# Install dependencies
uv pip install -r requirements.txt

# Run in paper trading mode (default)
python run.py

# Run main.py (minimal entry point)
python main.py

# Run API test script
bash test_api.sh
```

### Code Quality Checks
```bash
# Type checking (if mypy configured)
mypy core/ ai/ utils/ exchanges/

# Import sorting (if isort configured)
isort --check-only --diff core/ ai/ utils/ exchanges/

# Code formatting (if black configured)
black --check --diff core/ ai/ utils/ exchanges/
```

## Code Style Guidelines

### Python Standards
- **Python Version**: 3.12+
- **Type Hints**: Required for all function parameters and return values
- **Docstrings**: Use Google-style docstrings for all public functions
- **Line Length**: 88 characters (Black default)
- **Imports**: Follow PEP 8, group by standard library, third-party, local

### Import Organization
```python
# Standard library imports
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Third-party imports
import pandas as pd
import pandas_ta as ta

# Local imports (absolute)
from exchanges.base import BaseExchange
from ai.prompt_builder import build_prompt
```

### Naming Conventions
- **Classes**: PascalCase (`TradingBot`, `BaseExchange`)
- **Functions/Methods**: snake_case (`get_current_price`, `run_cycle`)
- **Variables**: snake_case (`account_balance`, `current_position`)
- **Constants**: UPPER_SNAKE_CASE (`STOP_LOSS_PERCENT`)
- **Files**: snake_case (`risk_manager.py`, `prompt_builder.py`)

### Async/Await Patterns
```python
# Correct async function signature
async def get_account_balance(self, currency: str) -> float:
    """Get account balance for specified currency."""
    async with self.session.get(url) as response:
        data = await response.json()
        return data['balance']

# Avoid nested async calls when possible
# Good: Sequential operations
balance = await exchange.get_account_balance(config['CURRENCY'])
price = await exchange.get_current_price(config['SYMBOL'])

# Bad: Unnecessary async context
async def wrapper():
    return await some_async_func()
```

### Error Handling
```python
# Comprehensive error handling with specific exceptions
try:
    position = await exchange.get_pending_positions(symbol)
    if not position:
        logger.warning(f"No position found for {symbol}")
        return None
except ConnectionError as e:
    logger.error(f"Network error fetching position: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid data received: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error in position fetch: {e}")
    raise
```

### Logging Standards
```python
# Use appropriate log levels
logger = logging.getLogger(__name__)
app_logger = logging.getLogger("app")  # User-facing messages

# Info level for normal operations
app_logger.info(f"Position opened: {side} {quantity} @ ${price:,.2f}")

# Warning for recoverable issues
logger.warning(f"Failed to update position: {e}")

# Error for failures requiring attention
logger.error(f"Trade execution failed: {e}")

# Debug for detailed internal state
logger.debug(f"Calculated position size: {quantity}")
```

### Configuration Management
```python
# All configuration from environment variables via dotenv
# No hardcoded values except defaults
from dotenv import load_dotenv
import os

load_dotenv()

# Good: Environment-based configuration
API_KEY = os.getenv('EXCHANGE_API_KEY', '')
LEVERAGE = int(os.getenv('LEVERAGE', '20'))

# Bad: Hardcoded values
API_KEY = "my-secret-key"
LEVERAGE = 20
```

## Security Requirements

### API Key Handling
- Never commit API keys, secrets, or credentials to repository
- All sensitive data must be in `.env` file (gitignored)
- Use environment variables for all configuration
- Validate API key presence before live trading

### Financial Safety
- Always implement proper position size limits
- Include stop-loss and take-profit orders
- Validate all trade parameters before execution
- Implement atomic operations with rollback on failure
- Log all financial operations with sufficient detail

## Trading Logic Rules

### Core Principles (from .clinerules/rules.txt)
1. **No Third-Party Wrappers**: Interact directly with exchange APIs
2. **Real Data Only**: Always fetch live data, never use dummy values
3. **Environment Configuration**: All variables in .env file via dotenv
4. **No Hardcoded Values**: Every parameter configurable via environment

### Position Management
```python
# Atomic position operations with verification
async def execute_trade(self, decision: Dict[str, Any], current_price: float):
    """Execute trade with atomic verification and rollback."""
    original_state = self.current_position.copy() if self.current_position else None

    try:
        # Execute trade
        await self._open_position(decision, current_price)

        # Verify position exists
        verified = await self.exchange.get_pending_positions(self.symbol)
        if not verified:
            raise ValueError("Position verification failed")

        # Update local state only after verification
        self.current_position = verified

    except Exception as e:
        # Rollback on any failure
        await self._rollback_trade()
        if original_state:
            self.current_position = original_state
        raise
```

### Risk Management
- Implement position size limits based on account balance
- Use volatility-adjusted sizing when enabled
- Always set stop-loss and take-profit orders
- Validate risk-reward ratios before trading
- Include fee calculations in position sizing

## Testing Guidelines

### Unit Test Structure
```python
# Use pytest fixtures for common setup
@pytest.fixture
def sample_config():
    return {
        'SYMBOL': 'BTCUSDT',
        'CURRENCY': 'USDT',
        'LEVERAGE': 5,
        'FORWARD_TESTING': True
    }

def test_build_prompt_minimal_config(sample_config):
    prompt = build_prompt(
        timestamp=datetime.utcnow(),
        minutes_elapsed=10,
        account_balance=1000.0,
        equity=1000.0,
        open_positions=[],
        # ... other parameters
    )
    assert isinstance(prompt, str)
    assert 'BTC' in prompt
```

### Test Coverage Requirements
- Core trading logic: 90%+ coverage
- Exchange integrations: 80%+ coverage
- AI prompt building: 85%+ coverage
- Risk management: 95%+ coverage

## Code Review Checklist

### Before Committing
- [ ] All type hints present and correct
- [ ] Docstrings for public functions
- [ ] No hardcoded values (except sensible defaults)
- [ ] Environment variables used for configuration
- [ ] Proper error handling with specific exceptions
- [ ] Logging at appropriate levels
- [ ] Tests added/updated for new functionality
- [ ] No API keys or secrets in code
- [ ] Direct exchange API usage (no CCXT wrappers)
- [ ] Atomic operations with rollback mechanisms

### Performance Considerations
- Avoid synchronous I/O in async functions
- Use appropriate data structures (pandas for OHLCV data)
- Implement connection pooling for API calls
- Cache expensive computations when safe
- Monitor memory usage in long-running processes

## Architecture Patterns

### Exchange Interface
```python
class BaseExchange(ABC):
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        pass

    @abstractmethod
    async def open_position(self, symbol: str, side: str, size: str) -> Dict:
        """Open new position with validation."""
        pass
```

### AI Integration
- Structured prompt building with clear parameters
- Confidence scoring for trade decisions
- Response parsing with validation
- Error handling for API failures

### Risk Management
- Position size calculation with fee consideration
- Stop-loss/take-profit order management
- Daily PnL tracking
- Volatility adjustment when enabled

## Git Workflow

### Commit Messages
```
feat: add volatility-adjusted position sizing
fix: handle API timeout in position verification
refactor: simplify atomic trade execution logic
test: add unit tests for risk manager
docs: update AGENTS.md with new guidelines
```

### Branch Naming
- `feature/volatility-adjustment`
- `fix/api-timeout-handling`
- `refactor/atomic-operations`
- `test/risk-manager-coverage`

## Cursor Rules Integration

The following rules from `.clinerules/rules.txt` must be strictly followed:

1. **Simplest Solution**: Choose the simplest solution possible
2. **Direct API Calls**: Never use CCXT or third-party exchange wrappers
3. **Official SDKs**: Interact with official exchange APIs via SDKs or REST/WebSocket
4. **Real Data**: Always fetch real data from exchanges
5. **No Hardcoded Values**: All variables in .env via dotenv
6. **No Dummy Data**: Never use dummy balance in forward testing

## Common Pitfalls to Avoid

1. **Race Conditions**: Use asyncio.Lock for shared state
2. **Memory Leaks**: Clean up async tasks and connections
3. **API Rate Limits**: Implement proper backoff and retry logic
4. **Precision Errors**: Use appropriate decimal handling for financial calculations
5. **State Inconsistency**: Always verify exchange state matches local state
6. **Unbounded Operations**: Implement timeouts for all network operations
7. **Log Pollution**: Use appropriate log levels, avoid excessive debug logging in production</content>
<parameter name="filePath">/home/y/MY_PROJECTS/TradingBots/ai-crypto-trader/AGENTS.md