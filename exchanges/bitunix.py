# exchanges/bitunix.py
import requests
import logging
import hmac
import hashlib
import time
import json
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseExchange, Position
BASE_URL = "https://api.bitunix.com"
FUTURES_URL = f"{BASE_URL}/api/v1/futures"  # Adjust if API version changes


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def api_call(method: str, endpoint: str, params: Dict = None, signed: bool = False, api_key: str = None, api_secret: str = None) -> Dict:
    """Helper for authenticated/public API calls."""
    if params is None:
        params = {}
    params['timestamp'] = int(time.time() * 1000)  # Bitunix uses ms timestamp
    url = f"{FUTURES_URL}{endpoint}"
    if signed:
        if not api_key or not api_secret:
            raise ValueError("API key and secret required for signed calls")
        query_string = '&'.join(
            [f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    else:
        headers = {"Content-Type": "application/json"}
    if method.upper() == 'GET':
        response = requests.get(url, params=params, headers=headers)
    elif method.upper() == 'POST':
        response = requests.post(url, json=params, headers=headers)
    else:
        raise ValueError(f"Unsupported method: {method}")
    response.raise_for_status()
    return response.json()


class BitunixFutures(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        # Load markets if needed (public call)
        try:
            markets = api_call('GET', '/markets', signed=False)
            logging.info("[LIVE] Bitunix markets loaded")
        except Exception as e:
            logging.warning(f"[LIVE] Could not load markets: {e}")

    def get_current_price(self, symbol: str) -> float:
        data = api_call('GET', f'/ticker/price?symbol={symbol}', signed=False)
        # Assumes response format: {"data": {"price": "67000.00"}}
        return float(data['data']['price'])

    def get_account_balance(self, currency: str) -> float:
        data = api_call('GET', '/account/balance', signed=True,
                        api_key=self.api_key, api_secret=self.api_secret)
        for asset in data['data']['assets']:
            if asset['currency'] == currency:
                # Adjust key if needed (e.g., 'walletBalance')
                return float(asset['availableBalance'])
        return 0.0

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        data = api_call(
            'GET', f'/positions?symbol={symbol}', signed=True, api_key=self.api_key, api_secret=self.api_secret)
        positions = data.get('data', [])
        for pos in positions:
            size = float(pos['size'])
            if size > 0:
                return Position(
                    positionId=pos['positionId'],
                    side=pos['side'].upper(),  # 'BUY' or 'SELL'
                    size=size,
                    entry_price=float(pos['entryPrice']),
                    symbol=symbol
                )
        return None

    def set_leverage(self, symbol: str, leverage: int):
        params = {'symbol': symbol, 'leverage': leverage}
        api_call('POST', '/leverage', params=params, signed=True,
                 api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Leverage set to {leverage}x for {symbol}")

    def set_margin_mode(self, symbol: str, mode: str):
        margin_mode = 'ISOLATED' if mode.upper(
        ) == 'ISOLATED' else 'CROSSED'  # Bitunix uses these terms
        params = {'symbol': symbol, 'marginMode': margin_mode}
        api_call('POST', '/marginMode', params=params, signed=True,
                 api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Margin mode set to {mode} for {symbol}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int] = None):
        balance = self.get_account_balance('USDT')
        price = self.get_current_price(symbol)
        # Position sizing (same as your original)
        if size.endswith('%'):
            pct = float(size[:-1]) / 100
            usdt_value = balance * pct
        else:
            usdt_value = float(size)
        # For BTCUSDT perpetual, quantity = usdt_value / price (1 contract = 1 USD worth)
        # Adjust precision as per Bitunix (usually 0.001 step)
        quantity = round(usdt_value / price, 3)
        params = {
            'symbol': symbol,
            'side': side.lower(),  # 'buy' or 'sell'
            'type': 'MARKET',
            'quantity': quantity
        }
        if sl_pct:
            if side.lower() == 'buy':
                sl_price = price * (1 - sl_pct / 100)
            else:
                sl_price = price * (1 + sl_pct / 100)
            params['stopLoss'] = round(sl_price, 2)  # Bitunix SL param
        api_call('POST', '/order', params=params, signed=True,
                 api_key=self.api_key, api_secret=self.api_secret)
        logging.info(
            f"[LIVE] Opened {side.upper()} {quantity} @ ${price:,.2f} (SL: {sl_pct}%)")

    def flash_close_position(self, symbol: str):
        position = self.get_pending_positions(symbol)
        if not position:
            return
        close_side = 'sell' if position.side == 'BUY' else 'buy'
        params = {
            'symbol': symbol,
            'side': close_side,
            'type': 'MARKET',
            'quantity': position.size,
            'reduceOnly': True  # Ensures full close without increasing position
        }
        api_call('POST', '/order', params=params, signed=True,
                 api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Closed {position.side} position at market")

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 15):
        # Map timeframe: Bitunix uses '1m', '5m', '1h', etc.
        params = {'symbol': symbol, 'interval': timeframe, 'limit': limit}
        data = api_call('GET', '/klines', params=params,
                        signed=False)  # Public endpoint
        ohlcv = []
        for kline in data['data']:
            ohlcv.append([
                int(kline[0]),  # timestamp
                float(kline[1]),  # open
                float(kline[2]),  # high
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5])   # volume
            ])
        return ohlcv
