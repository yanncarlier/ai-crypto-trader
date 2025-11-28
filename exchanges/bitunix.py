# exchanges/bitunix.py
import ccxt
import logging
from typing import Optional
from .base import BaseExchange, Position


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
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    def get_account_balance(self, currency: str) -> float:
        balance = self.exchange.fetch_balance(params={'type': 'swap'})
        return balance['total'].get(currency, 0)

    def get_pending_positions(self, symbol: str):
        positions = self.exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['contracts'] > 0:
                return Position(pos['info']['positionId'], pos['side'], pos['contracts'], pos['entryPrice'])
        return None

    def set_leverage(self, symbol: str, leverage: int):
        self.exchange.set_leverage(leverage, symbol)

    def set_margin_mode(self, symbol: str, mode: str):
        self.exchange.set_margin_mode(mode.lower(), symbol)

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        # size can be percentage or fixed
        balance = self.get_account_balance('USDT')
        if size.endswith('%'):
            pct = int(size[:-1]) / 100
            usdt_amount = balance * pct
        else:
            usdt_amount = float(size)
        price = self.get_current_price(symbol)
        amount = usdt_amount / price * \
            self.exchange.market(symbol)['contractSize']
        params = {'positionIdx': 0}
        if sl_pct:
            sl_price = price * \
                (1 - sl_pct/100) if side == 'sell' else price * (1 + sl_pct/100)
            params['stopLoss'] = str(round(sl_price, 2))
        order = self.exchange.create_order(
            symbol, 'market', side, amount, None, params)
        logging.info(f"LIVE: Opened {side.upper()} {amount:.4f} contracts")

    def flash_close_position(self, position_id: str):
        # Bitunix doesn't expose positionId directly in close, so close by symbol
        position = self.get_pending_positions(self.exchange.symbols[0])
        if position:
            side = 'sell' if position.side == 'BUY' else 'buy'
            self.exchange.create_order(
                position.symbol, 'market', side, position.size)
            logging.info("LIVE: Position closed at market")
