# exchanges/bitunix.py
import json
import hashlib
import time
import secrets
import logging
from typing import Optional, Dict, Any
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseExchange, Position
API_URL = "https://fapi.bitunix.com/api/v1/futures"
TIMEOUT = 10


class BitunixAuth:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key

    def _generate_signature(self, nonce: str, timestamp: str, query_params: str = "", body: str = "") -> str:
        digest_input = f"{nonce}{timestamp}{self.api_key}{query_params}{body}"
        digest = hashlib.sha256(digest_input.encode()).hexdigest()
        return hashlib.sha256(f"{digest}{self.secret_key}".encode()).hexdigest()

    def get_headers(self, query_params: str = "", body: str = "") -> Dict[str, str]:
        nonce = secrets.token_hex(16)
        timestamp = str(int(time.time() * 1000))
        return {
            "api-key": self.api_key,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": self._generate_signature(nonce, timestamp, query_params, body),
            "Content-Type": "application/json"
        }


class BitunixFutures(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):
        self.auth = BitunixAuth(api_key, api_secret)
        logging.info("[LIVE] Bitunix Futures client initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _public_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{API_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise Exception(
                f"API Error {result.get('code')}: {result.get('msg')}")
        return result["data"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{API_URL}{endpoint}"
        sorted_params = ""
        if params:
            sorted_items = sorted(params.items(), key=lambda x: x[0])
            sorted_params = "".join(f"{k}{v}" for k, v in sorted_items)
        headers = self.auth.get_headers(query_params=sorted_params)
        response = requests.get(url, headers=headers,
                                params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise Exception(f"API Error {data.get('code')}: {data.get('msg')}")
        return data["data"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Any:
        url = f"{API_URL}{endpoint}"
        payload = json.dumps(data, separators=(',', ':'))
        headers = self.auth.get_headers(body=payload)
        response = requests.post(url, headers=headers,
                                 data=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise Exception(
                f"API Error {result.get('code')}: {result.get('msg')}")
        return result["data"]

    def get_current_price(self, symbol: str) -> float:
        data = self._public_get("/market/tickers", {"symbols": symbol})
        for t in data:
            if t["symbol"] == symbol:
                return float(t["lastPrice"])
        raise ValueError(f"Price not found for {symbol}")

    def get_account_balance(self, currency: str) -> float:
        data = self._get("/account", {"marginCoin": currency})
        return float(data["margin"])

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        data = self._get("/position/get_pending_positions", {"symbol": symbol})
        if not data:
            return None
        pos = data[0]
        side = "BUY" if float(pos["qty"]) > 0 else "SELL"
        return Position(
            positionId=pos["positionId"],
            side=side,
            size=abs(float(pos["qty"])),
            entry_price=float(pos["avgOpenPrice"]),
            symbol=symbol
        )

    def set_leverage(self, symbol: str, leverage: int):
        self._post("/account/change_leverage", {
            "symbol": symbol,
            "leverage": leverage,
            "marginCoin": "USDT"
        })
        logging.info(f"[LIVE] Leverage set to {leverage}x")

    def set_margin_mode(self, symbol: str, mode: str):
        mode_str = "ISOLATION" if mode.upper() == "ISOLATED" else "CROSS"
        self._post("/account/change_margin_mode", {
            "symbol": symbol,
            "marginMode": mode_str,
            "marginCoin": "USDT"
        })
        logging.info(f"[LIVE] Margin mode: {mode_str}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        balance = self.get_account_balance("USDT")
        price = self.get_current_price(symbol)
        if size.endswith("%"):
            usdt_value = balance * (float(size[:-1]) / 100)
        else:
            usdt_value = float(size)
        # Calculate raw BTC quantity
        qty = usdt_value / price
        # ENFORCE MINIMUM SIZE: Bitunix requires >= 0.0001 BTC
        MIN_QTY = 0.0001
        if qty < MIN_QTY:
            if balance * 0.1 < MIN_QTY * price:  # If even 10% of balance is too small
                logging.warning(
                    f"[LIVE] Balance too low (${balance:,.2f}) to open minimum position. Skipping trade.")
                return
            else:
                logging.info(
                    f"[LIVE] Position size too small ({qty:.6f} BTC), forcing minimum {MIN_QTY} BTC")
                qty = MIN_QTY
        # Round to 4 decimals (Bitunix allows up to 0.0001 precision)
        qty = round(qty, 4)
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.upper(),
            "tradeSide": "OPEN",
            "orderType": "MARKET",
            "marginCoin": "USDT"
        }
        if sl_pct:
            sl_price = price * \
                (1 - sl_pct/100) if side.lower() == "buy" else price * (1 + sl_pct/100)
            order_data.update({
                "slPrice": str(round(sl_price, 2)),
                "slStopType": "MARK_PRICE",
                "slOrderType": "MARKET"
            })
        self._post("/trade/place_order", order_data)
        sl_text = f" | SL {sl_pct}%" if sl_pct else ""
        logging.info(
            f"[LIVE] Opened {side.upper()} {qty} BTC (${usdt_value:,.1f}) @ ${price:,.2f}{sl_text}")

    def flash_close_position(self, symbol: str):
        position = self.get_pending_positions(symbol)
        if not position:
            logging.info("[LIVE] No open position to close")
            return
        self._post("/trade/flash_close_position", {
            "positionId": position.positionId
        })
        logging.info(
            f"[LIVE] Flash closed {position.side} position ({position.size} BTC)")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 15):
        data = self._public_get("/market/kline", {
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit
        })
        ohlcv = []
        for k in data:
            ohlcv.append([
                k["time"],           # timestamp
                float(k["open"]),    # open
                float(k["high"]),    # high
                float(k["low"]),     # low
                float(k["close"]),   # close
                # volume (USDT amount; use "baseVol" if you prefer BTC volume)
                float(k["quoteVol"])
            ])
        return ohlcv
