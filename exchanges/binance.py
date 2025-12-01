# exchanges/binance.py
import ccxt
import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from exchanges.base import BaseExchange, Position


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError))
)
def safe_ccxt_call(func, *args, **kwargs):
    return func(*args, **kwargs)


class BinanceFutures(BaseExchange):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
            'enableRateLimit': True,
        }
        if testnet:
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }
            logging.info("[LIVE] Binance Futures Testnet client initialized")
        else:
            logging.info("[LIVE] Binance Futures Live client initialized")
        self.exchange = ccxt.binance(config)
        self.exchange.load_markets()

    def get_current_price(self, symbol: str) -> float:
        ticker = safe_ccxt_call(self.exchange.fetch_ticker, symbol)
        return float(ticker['last'])

    def get_account_balance(self, currency: str) -> float:
        balance = safe_ccxt_call(
            self.exchange.fetch_balance, params={'type': 'future'})
        return float(balance['total'].get(currency, 0.0))

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        positions = safe_ccxt_call(self.exchange.fetch_positions, [
                                   symbol], params={'type': 'future'})
        for pos in positions:
            if pos['contracts'] and float(pos['contracts']) > 0:
                return Position(
                    positionId=pos['info']['positionId'],
                    side=pos['side'].upper(),
                    size=float(pos['contracts']),
                    entry_price=float(pos['entryPrice']),
                    symbol=symbol
                )
        return None

    def set_leverage(self, symbol: str, leverage: int):
        safe_ccxt_call(self.exchange.set_leverage, leverage,
                       symbol, params={'type': 'future'})
        logging.info(f"[LIVE] Leverage set to {leverage}x")

    def set_margin_mode(self, symbol: str, mode: str):
        margin_mode = 'isolated' if mode.lower() == 'isolated' else 'cross'
        safe_ccxt_call(self.exchange.set_margin_mode,
                       margin_mode, symbol, params={'type': 'future'})
        logging.info(f"[LIVE] Margin mode: {mode}")

    def get_position_size(self, symbol: str, balance: float, size_config: str) -> float:
        """Calculate position size based on configuration"""
        if size_config.endswith('%'):
            percentage = float(size_config[:-1]) / 100
            return balance * percentage
        else:
            return float(size_config)

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        balance = self.get_account_balance('USDT')
        price = self.get_current_price(symbol)
        market = self.exchange.market(symbol)
        # Use new position size calculation
        usdt_value = self.get_position_size(symbol, balance, size)
        contract_size = market['contractSize'] or 1
        contracts = usdt_value / (price * contract_size)
        contracts = self.exchange.amount_to_precision(symbol, contracts)
        params = {'type': 'future'}
        if sl_pct:
            sl_price = price * \
                (1 - sl_pct/100) if side == 'buy' else price * (1 + sl_pct/100)
            params['stopLossPrice'] = self.exchange.price_to_precision(
                symbol, sl_price)
        order = safe_ccxt_call(
            self.exchange.create_order,
            symbol, 'market', side, contracts, None, params
        )
        sl_text = f" | SL {sl_pct}%" if sl_pct else ""
        logging.info(
            f"[LIVE] Opened {side.upper()} {contracts} contracts @ ${price:,.2f}{sl_text}")

    def flash_close_position(self, symbol: str):
        position = self.get_pending_positions(symbol)
        if not position:
            logging.info("[LIVE] No open position to close")
            return
        side = 'sell' if position.side == 'BUY' else 'buy'
        safe_ccxt_call(
            self.exchange.create_order,
            symbol, 'market', side, position.size, params={'type': 'future'}
        )
        logging.info(f"[LIVE] Closed {position.side} position at market")

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 15):
        return safe_ccxt_call(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit, params={'type': 'future'})
