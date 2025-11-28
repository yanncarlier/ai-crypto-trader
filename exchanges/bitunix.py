# exchanges/bitunix.py
import ccxt
import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseExchange, Position


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_ccxt_call(func, *args, **kwargs):
    return func(*args, **kwargs)


class BitunixFutures(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.bitunix({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'swap'},
            'enableRateLimit': True,
        })
        self.exchange.load_markets()

    def get_current_price(self, symbol: str) -> float:
        ticker = safe_ccxt_call(self.exchange.fetch_ticker, symbol)
        return float(ticker['last'])

    def get_account_balance(self, currency: str) -> float:
        balance = safe_ccxt_call(
            self.exchange.fetch_balance, params={'type': 'swap'})
        return float(balance['total'].get(currency, 0.0))

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        positions = safe_ccxt_call(self.exchange.fetch_positions, [symbol])
        for pos in positions:
            if pos['contracts'] > 0:
                return Position(
                    positionId=pos['info']['positionId'],
                    side=pos['side'].upper(),
                    size=float(pos['contracts']),
                    entry_price=float(pos['entryPrice']),
                    symbol=symbol
                )
        return None

    def set_leverage(self, symbol: str, leverage: int):
        safe_ccxt_call(self.exchange.set_leverage, leverage, symbol)
        logging.info(f"[LIVE] Leverage set to {leverage}x")

    def set_margin_mode(self, symbol: str, mode: str):
        safe_ccxt_call(self.exchange.set_margin_mode, mode.lower(), symbol)
        logging.info(f"[LIVE] Margin mode: {mode}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        balance = self.get_account_balance('USDT')
        price = self.get_current_price(symbol)
        market = self.exchange.market(symbol)
        contract_size = market['contractSize']
        if size.endswith('%'):
            pct = float(size[:-1]) / 100
            usdt_value = balance * pct
        else:
            usdt_value = float(size)
        contracts = usdt_value / (price * contract_size)
        contracts = round(contracts, 3)  # Adjust precision as needed
        params = {}
        if sl_pct:
            sl_price = price * \
                (1 - sl_pct/100) if side == 'sell' else price * (1 + sl_pct/100)
            params['stopLoss'] = str(round(sl_price, 2))
        order = safe_ccxt_call(
            self.exchange.create_order,
            symbol, 'market', side, contracts, None, params
        )
        logging.info(
            f"[LIVE] Opened {side.upper()} {contracts} contracts @ ${price:,.2f}")

    def flash_close_position(self, symbol: str):
        position = self.get_pending_positions(symbol)
        if not position:
            return
        side = 'sell' if position.side == 'BUY' else 'buy'
        safe_ccxt_call(
            self.exchange.create_order,
            symbol, 'market', side, position.size
        )
        logging.info(f"[LIVE] Closed {position.side} position at market")
