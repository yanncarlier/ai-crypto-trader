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
BASE_URL = "https://api-v1.bitunix.com"  # Updated base URL
FUTURES_PATH = "/fapi/v1"  # Futures prefix


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def api_call(method: str, endpoint: str, params: Dict = None, signed: bool = False, api_key: str = None, api_secret: str = None) -> Dict:
    """Helper for authenticated/public API calls."""
    if params is None:
        params = {}
    params['timestamp'] = int(time.time() * 1000)  # ms timestamp
    url = f"{BASE_URL}{FUTURES_PATH}{endpoint}"
    if signed:
        if not api_key or not api_secret:
            raise ValueError("API key and secret required for signed calls")
        # For signed POST, use body as JSON; sign query + body string
        if method.upper() == 'POST':
            body = json.dumps(params, separators=(',', ':'))
            sign_string = f"{json.dumps({k: v for k, v in sorted(params.items()) if k != 'signature'})}|{body}"
        else:
            query_string = '&'.join(
                [f"{k}={v}" for k, v in sorted(params.items()) if k != 'signature'])
            sign_string = query_string
        signature = hmac.new(
            api_secret.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest().upper()  # Bitunix often uppercases sig
        params['signature'] = signature
        headers = {
            "api-key": api_key,  # Correct header key
            "Content-Type": "application/json"
        }
    else:
        headers = {"Content-Type": "application/json"}
    try:
        if method.upper() == 'GET':
            response = requests.get(
                url, params=params, headers=headers, timeout=10)
        elif method.upper() == 'POST':
            response = requests.post(
                url, json=params, headers=headers, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        response.raise_for_status()
        data = response.json()
        # Handle common response formats
        if 'code' in data and data['code'] != 200:
            raise ValueError(f"API error: {data.get('msg', 'Unknown')}")
        if not data.get('success', True):
            raise ValueError(f"API failed: {data}")
        return data
    except requests.exceptions.HTTPError as e:
        logging.error(
            f"HTTP Error {e.response.status_code} for {url}: {e.response.text[:200]}")
        raise
    except Exception as e:
        logging.error(f"API call failed for {url}: {e}")
        raise


class BitunixFutures(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        # Try to load markets (public)
        try:
            # Common endpoint for symbols
            markets = api_call('GET', '/exchangeInfo', signed=False)
            logging.info(
                f"[LIVE] Bitunix markets loaded: {len(markets.get('data', {}).get('symbols', []))} symbols")
        except Exception as e:
            logging.warning(f"[LIVE] Could not load markets: {e}")

    def get_current_price(self, symbol: str) -> float:
        data = api_call('GET', f'/ticker/price?symbol={symbol}', signed=False)
        price_key = data.get('data', {}).get('price') or data.get('price')
        return float(price_key)

    def get_account_balance(self, currency: str) -> float:
        data = api_call('GET', '/account', signed=True, api_key=self.api_key,
                        api_secret=self.api_secret)  # /account for futures balance
        assets = data.get('data', {}).get('assets', [])
        for asset in assets:
            if asset.get('asset') == currency:
                return float(asset.get('availableBalance', 0))
        return 0.0

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        data = api_call(
            'GET', f'/positionRisk?symbol={symbol}', signed=True, api_key=self.api_key, api_secret=self.api_secret)
        positions = data.get('data', [])
        for pos in positions:
            size = float(pos.get('positionAmt', 0))
            if size != 0:  # >0 or <0, but abs for size
                return Position(
                    positionId=pos.get('positionId', 'unknown'),
                    # Adjust based on response
                    side='BUY' if float(
                        pos.get('positionSide', '')) == 1 else 'SELL',
                    size=abs(size),
                    entry_price=float(pos.get('entryPrice', 0)),
                    symbol=symbol
                )
        return None

    def set_leverage(self, symbol: str, leverage: int):
        params = {'symbol': symbol, 'leverage': leverage,
                  'marginType': 'ISOLATED'}
        api_call('POST', '/leverage', params=params, signed=True,
                 api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Leverage set to {leverage}x for {symbol}")

    def set_margin_mode(self, symbol: str, mode: str):
        margin_mode = 'ISOLATED' if mode.upper() == 'ISOLATED' else 'CROSS'
        params = {'symbol': symbol, 'marginType': margin_mode}
        api_call('POST', '/marginType', params=params, signed=True,
                 api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Margin mode set to {mode} for {symbol}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int] = None):
        balance = self.get_account_balance('USDT')
        price = self.get_current_price(symbol)
        # Position sizing
        if size.endswith('%'):
            pct = float(size[:-1]) / 100
            usdt_value = balance * pct
        else:
            usdt_value = float(size)
        quantity = round(usdt_value / price, 6)  # Higher precision for BTCUSDT
        params = {
            'symbol': symbol,
            'side': side.lower(),
            'type': 'MARKET',
            'quantity': quantity
        }
        if sl_pct:
            sl_price = price * \
                (1 - sl_pct / 100) if side.lower() == 'buy' else price * \
                (1 + sl_pct / 100)
            params['stopPrice'] = round(sl_price, 2)  # Or 'stopLossPrice'
            # Adjust for SL order
            params['type'] = 'STOP_MARKET' if sl_pct else 'MARKET'
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
            'reduceOnly': True
        }
        api_call('POST', '/order', params=params, signed=True,
                 api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Closed {position.side} position at market")

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 15):
        params = {'symbol': symbol, 'interval': timeframe, 'limit': limit}
        data = api_call('GET', '/klines', params=params, signed=False)
        ohlcv = []
        klines = data.get('data', [])
        for kline in klines:
            ohlcv.append([
                int(kline[0]),  # timestamp
                float(kline[1]),  # open
                float(kline[2]),  # high
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5])   # volume
            ])
        return ohlcv
