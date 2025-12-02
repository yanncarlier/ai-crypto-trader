# exchanges/bitunix.py
import json
import hashlib
import time
import secrets
import logging
from typing import Optional, Dict, Any, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from exchanges.base import BaseExchange, Position
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
            "Content-Type": "application/json",
        }


class BitunixFutures(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):
        self.auth = BitunixAuth(api_key, api_secret)
        self.last_request_time = 0
        self.min_request_interval = 0.1
        self.symbol = "BTCUSDT"  # Default symbol

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _public_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._rate_limit()
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
        self._rate_limit()
        url = f"{API_URL}{endpoint}"
        sorted_params = "".join(f"{k}{v}" for k, v in sorted(
            (params or {}).items())) if params else ""
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
        self._rate_limit()
        url = f"{API_URL}{endpoint}"
        payload = json.dumps(data, separators=(",", ":"))
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
                price = float(t["lastPrice"])
                return price
        raise ValueError(f"Price not found for {symbol}")

    def get_account_balance(self, currency: str) -> float:
        """Get total account balance in USDT (including BTC converted to USDT)"""
        try:
            # Get USDT balance
            data_usdt = self._get("/account", {"marginCoin": "USDT"})
            usdt_balance = float(data_usdt.get("available") or 0.0)
            total_usdt = float(data_usdt.get("total") or 0.0)
            # Get BTC balance
            data_btc = self._get("/account", {"marginCoin": "BTC"})
            btc_balance = float(data_btc.get("available") or 0.0)
            total_btc = float(data_btc.get("total") or 0.0)
            # Get current BTC price to convert to USDT
            btc_price = self.get_current_price("BTCUSDT")
            btc_in_usdt = btc_balance * btc_price
            total_btc_in_usdt = total_btc * btc_price
            # Total available balance = USDT + BTC converted to USDT
            total_available = usdt_balance + btc_in_usdt
            total_equity = total_usdt + total_btc_in_usdt
            # Log detailed balance
            logging.info(f"💰 USDT Balance: ${usdt_balance:,.2f} available")
            if btc_balance > 0:
                logging.info(
                    f"💰 BTC Balance: {btc_balance:.8f} BTC (≈${btc_in_usdt:,.2f})")
            logging.info(f"📊 Total Available: ${total_available:,.2f}")
            logging.info(f"📈 Total Equity: ${total_equity:,.2f}")
            return total_available  # Return total available in USDT
        except Exception as e:
            logging.error(f"❌ Failed to fetch balance: {e}")
            return 0.0

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        try:
            data = self._get("/position/get_pending_positions",
                             {"symbol": symbol})
            if not data:
                return None
            for pos in data:
                if pos.get("symbol") == symbol and float(pos.get("qty", 0)) != 0:
                    side = "BUY" if float(pos["qty"]) > 0 else "SELL"
                    position = Position(
                        positionId=pos["positionId"],
                        side=side,
                        size=abs(float(pos["qty"])),
                        entry_price=float(pos["avgOpenPrice"]),
                        symbol=symbol,
                    )
                    # Calculate position details
                    current_price = self.get_current_price(symbol)
                    position_value = position.size * current_price
                    pnl = (current_price - position.entry_price) * \
                        position.size
                    pnl_pct = ((current_price - position.entry_price) /
                               position.entry_price) * 100
                    logging.info(f"📦 Existing {side} position:")
                    logging.info(f"   Size: {position.size:.4f} BTC")
                    logging.info(f"   Entry: ${position.entry_price:,.2f}")
                    logging.info(f"   Current: ${current_price:,.2f}")
                    logging.info(f"   PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
                    logging.info(f"   Value: ${position_value:,.2f}")
                    return position
        except Exception as e:
            logging.warning(f"Failed to fetch positions: {e}")
        return None

    def get_all_positions(self, symbol: str) -> List[Position]:
        """Get all positions for the symbol (should only be one for futures)"""
        try:
            data = self._get("/position/get_pending_positions",
                             {"symbol": symbol})
            positions = []
            for pos in data:
                if float(pos.get("qty", 0)) != 0:
                    side = "BUY" if float(pos["qty"]) > 0 else "SELL"
                    positions.append(Position(
                        positionId=pos["positionId"],
                        side=side,
                        size=abs(float(pos["qty"])),
                        entry_price=float(pos["avgOpenPrice"]),
                        symbol=symbol,
                    ))
            return positions
        except Exception as e:
            logging.warning(f"Failed to fetch all positions: {e}")
            return []

    def get_account_summary(self, currency: str, symbol: str) -> Dict[str, Any]:
        """Get complete account summary including balance and positions"""
        balance = self.get_account_balance(currency)
        positions = self.get_all_positions(symbol)
        summary = {
            "balance": balance,
            "currency": currency,
            "positions": positions,
            "total_positions": len(positions),
            "total_exposure": 0.0,
        }
        if positions:
            current_price = self.get_current_price(symbol)
            for position in positions:
                position_value = position.size * current_price
                summary["total_exposure"] += position_value
                # Calculate PnL for each position
                pnl = (current_price - position.entry_price) * position.size
                position.pnl = pnl
                position.pnl_percent = (
                    (current_price - position.entry_price) / position.entry_price) * 100
        return summary

    def set_leverage(self, symbol: str, leverage: int):
        self._post("/account/change_leverage",
                   {"symbol": symbol, "leverage": leverage, "marginCoin": "USDT"})

    def set_margin_mode(self, symbol: str, mode: str):
        mode_str = "ISOLATION" if mode.upper() == "ISOLATED" else "CROSS"
        self._post("/account/change_margin_mode",
                   {"symbol": symbol, "marginMode": mode_str, "marginCoin": "USDT"})

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        # Get account summary first
        summary = self.get_account_summary("USDT", symbol)
        balance = summary["balance"]
        existing_position = summary["positions"][0] if summary["positions"] else None
        price = self.get_current_price(symbol)
        # Calculate position size considering existing position
        if size.endswith("%"):
            percentage = float(size[:-1]) / 100
            if existing_position and existing_position.side.lower() == side.lower():
                # If we're adding to existing position in same direction
                current_position_value = existing_position.size * price
                max_additional_value = balance * percentage
                total_desired_value = current_position_value + max_additional_value
                # Check if we're exceeding available balance
                if max_additional_value > balance:
                    logging.warning(
                        f"Not enough balance to add to position. Available: ${balance:,.2f}")
                    return
                additional_qty = max_additional_value / price
                total_qty = existing_position.size + additional_qty
                logging.info(f"📈 Adding to existing {side} position:")
                logging.info(f"   Current: {existing_position.size:.4f} BTC")
                logging.info(f"   Adding: {additional_qty:.4f} BTC")
                logging.info(f"   New total: {total_qty:.4f} BTC")
                # We need to close existing and open new combined position
                if existing_position:
                    self.flash_close_position(symbol)
                # Open new combined position
                usdt_value = total_desired_value
                qty = total_qty
            else:
                # New position or reversing
                usdt_value = balance * percentage
                qty = usdt_value / price
        else:
            usdt_value = float(size)
            qty = usdt_value / price
        # Validate minimum quantity
        MIN_QTY = 0.0001
        if qty < MIN_QTY:
            logging.warning(
                f"Position size too small ({qty:.6f} BTC). Minimum is {MIN_QTY} BTC")
            return
        qty = round(qty, 4)
        # Prepare order
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.upper(),
            "tradeSide": "OPEN",
            "orderType": "MARKET",
            "marginCoin": "USDT",
        }
        # Add stop loss if specified
        if sl_pct:
            if side.lower() == "buy":
                sl_price = price * (1 - sl_pct / 100)
            else:
                sl_price = price * (1 + sl_pct / 100)
            order_data.update({
                "slPrice": str(round(sl_price, 2)),
                "slStopType": "MARK_PRICE",
                "slOrderType": "MARKET",
            })
        # Execute trade
        self._post("/trade/place_order", order_data)
        # Log trade details
        action = "Added to" if existing_position and existing_position.side.lower(
        ) == side.lower() else "Opened"
        sl_text = f" | SL {sl_pct}%" if sl_pct else ""
        logging.info(
            f"✅ {action} {side.upper()} {qty:.4f} BTC (~${usdt_value:,.1f}) @ ${price:,.2f}{sl_text}")

    def flash_close_position(self, symbol: str):
        position = self.get_pending_positions(symbol)
        if not position:
            logging.info("No open position to close")
            return
        # Calculate PnL before closing
        current_price = self.get_current_price(symbol)
        pnl = (current_price - position.entry_price) * position.size
        pnl_pct = ((current_price - position.entry_price) /
                   position.entry_price) * 100
        # Close position
        self._post("/trade/flash_close_position",
                   {"positionId": position.positionId})
        logging.info(f"🔒 Closed {position.side} position:")
        logging.info(f"   Size: {position.size:.4f} BTC")
        logging.info(f"   Entry: ${position.entry_price:,.2f}")
        logging.info(f"   Exit: ${current_price:,.2f}")
        logging.info(f"   PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 15):
        data = self._public_get(
            "/market/kline", {"symbol": symbol, "interval": timeframe, "limit": limit})
        ohlcv = []
        for k in data:
            ohlcv.append([
                k["time"],
                float(k["open"]),
                float(k["high"]),
                float(k["low"]),
                float(k["close"]),
                float(k["quoteVol"]),
            ])
        return ohlcv
