# exchanges/bitunix.py
import requests
import logging
import hmac
import hashlib
import time
import json
import uuid
from typing import Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseExchange, Position
BASE_URL = "https://fapi.bitunix.com"  # Correct futures base URL


def generate_signature(api_key: str, secret_key: str, nonce: str, timestamp: str, query_string: str, body: str = "") -> str:
    """Bitunix double SHA256 signature."""
    digest_input = nonce + timestamp + api_key + query_string + body
    first_hash = hashlib.sha256(digest_input.encode('utf-8')).hexdigest()
    signature = hashlib.sha256(
        (first_hash + secret_key).encode('utf-8')).hexdigest()
    return signature


def build_query_string(params: Dict) -> str:
    """Build Bitunix query string: sorted keyvalue without separators."""
    sorted_params = sorted(params.items())
    return ''.join(f"{k}{v}" for k, v in sorted_params)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def api_call(method: str, endpoint: str, params: Dict = None, signed: bool = False, api_key: str = None, api_secret: str = None) -> Dict:
    """Helper for API calls."""
    if params is None:
        params = {}
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    nonce = uuid.uuid4().hex  # 32-char nonce
    query_string = build_query_string(params) if params else ""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    body = ""
    if signed:
        if not api_key or not api_secret:
            raise ValueError("API key and secret required for signed calls")
        if method.upper() == 'POST':
            body_json = json.dumps(params, separators=(',', ':'))
            body = body_json
            sign_string = query_string + body  # For POST: query + body
        else:
            sign_string = query_string
        signature = generate_signature(
            api_key, api_secret, nonce, timestamp_str, sign_string, body)
        headers.update({
            "api-key": api_key,
            "sign": signature,
            "timestamp": timestamp_str,
            "nonce": nonce
        })
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
        if data.get('code') != 0:
            raise ValueError(
                f"API error {data.get('code')}: {data.get('msg', 'Unknown')}")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Error for {url}: {e}")
        raise
    except Exception as e:
        logging.error(f"API call failed for {url}: {e}")
        raise


class BitunixFutures(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        # Load markets (public)
        try:
            markets = api_call(
                'GET', '/api/v1/futures/market/trading_pairs', signed=False)
            logging.info(
                f"[LIVE] Bitunix markets loaded: {len(markets.get('data', []))} pairs")
        except Exception as e:
            logging.warning(f"[LIVE] Could not load markets: {e}")

    def get_current_price(self, symbol: str) -> float:
        params = {'symbol': symbol}
        data = api_call('GET', '/api/v1/futures/market/ticker',
                        params=params, signed=False)
        return float(data['data'][0]['lastPrice'])  # Assumes array response

    def get_account_balance(self, currency: str) -> float:
        params = {'marginCoin': currency}
        data = api_call('GET', '/api/v1/futures/account/singleAccount', params=params,
                        signed=True, api_key=self.api_key, api_secret=self.api_secret)
        return float(data['data']['availableMargin'])

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        params = {'symbol': symbol}
        data = api_call('GET', '/api/v1/futures/position/list', params=params,
                        signed=True, api_key=self.api_key, api_secret=self.api_secret)
        positions = data.get('data', [])
        for pos in positions:
            size = float(pos.get('positionAmt', 0))
            if size != 0:
                return Position(
                    positionId=pos.get('positionId', 'unknown'),
                    side='BUY' if size > 0 else 'SELL',
                    size=abs(size),
                    entry_price=float(pos.get('entryPrice', 0)),
                    symbol=symbol
                )
        return None

    def set_leverage(self, symbol: str, leverage: int):
        params = {'symbol': symbol, 'marginCoin': 'USDT', 'leverage': leverage}
        api_call('POST', '/api/v1/futures/account/change_leverage', params=params,
                 signed=True, api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Leverage set to {leverage}x for {symbol}")

    def set_margin_mode(self, symbol: str, mode: str):
        margin_mode = 'ISOLATED_MARGIN' if mode.upper() == 'ISOLATED' else 'CROSS_MARGIN'
        params = {'symbol': symbol, 'marginCoin': 'USDT',
                  'marginMode': margin_mode}
        api_call('POST', '/api/v1/futures/account/change_margin_mode', params=params,
                 signed=True, api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Margin mode set to {mode} for {symbol}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int] = None):
        balance = self.get_account_balance('USDT')
        price = self.get_current_price(symbol)
        # Position sizing (USDT value / price for BTC qty)
        if size.endswith('%'):
            pct = float(size[:-1]) / 100
            usdt_value = balance * pct
        else:
            usdt_value = float(size)
        quantity = round(usdt_value / price, 3)  # BTC precision for BTCUSDT
        params = {
            'symbol': symbol,
            'marginCoin': 'USDT',
            'side': side.lower(),
            'orderType': 'MARKET',
            'size': quantity
        }
        if sl_pct:
            sl_price = price * \
                (1 - sl_pct / 100) if side.lower() == 'buy' else price * \
                (1 + sl_pct / 100)
            params['stopLossPrice'] = round(sl_price, 2)
        api_call('POST', '/api/v1/futures/order/create', params=params,
                 signed=True, api_key=self.api_key, api_secret=self.api_secret)
        logging.info(
            f"[LIVE] Opened {side.upper()} {quantity} @ ${price:,.2f} (SL: {sl_pct}%)")

    def flash_close_position(self, symbol: str):
        position = self.get_pending_positions(symbol)
        if not position:
            return
        close_side = 'sell' if position.side == 'BUY' else 'buy'
        params = {
            'symbol': symbol,
            'marginCoin': 'USDT',
            'side': close_side,
            'orderType': 'MARKET',
            'size': position.size,
            'reduceOnly': True
        }
        api_call('POST', '/api/v1/futures/order/create', params=params,
                 signed=True, api_key=self.api_key, api_secret=self.api_secret)
        logging.info(f"[LIVE] Closed {position.side} position at market")

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 15):
        params = {'symbol': symbol, 'interval': timeframe, 'limit': limit}
        data = api_call('GET', '/api/v1/futures/market/kline',
                        params=params, signed=False)
        ohlcv = []
        klines = data.get('data', [])
        for kline in klines:
            ohlcv.append([
                int(kline[0]),  # timestamp (ms)
                float(kline[1]),  # open
                float(kline[2]),  # high
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5])   # volume
            ])
        return ohlcv
