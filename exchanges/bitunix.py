# exchanges/bitunix.py
import json
import hashlib
import time
import secrets
import logging
import asyncio
import traceback
from typing import Optional, Dict, Any, List, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from exchanges.base import BaseExchange, Position
from utils.risk_manager import RiskManager

API_URL = "https://fapi.bitunix.com"
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
    def __init__(self, api_key: str, api_secret: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Bitunix Futures exchange client.
        
        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            config: Configuration dictionary (optional)
        """
        self.auth = BitunixAuth(api_key, api_secret)
        self.config = config or {}
        self.last_request_time = 0
        self.min_request_interval = 0.1
        self.symbol = self.config.get('SYMBOL', 'BTCUSDT')
        self.risk_manager = RiskManager(self.config, self) if self.config else None
        self._monitor_task = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _public_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        async with httpx.AsyncClient() as client:
            url = f"{API_URL}{endpoint}"
            response = await client.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            if result is None:
                raise Exception("Invalid JSON response from public API")
            if result.get("code") != 0:
                raise Exception(
                    f"API Error {result.get('code')}: {result.get('msg')}")
            return result["data"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        # Always fetch real data even in forward testing mode
        url = f"{API_URL}{endpoint}"
        sorted_params = "".join(f"{k}{v}" for k, v in sorted(
            (params or {}).items())) if params else ""
        headers = self.auth.get_headers(query_params=sorted_params)
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers,
                                    params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data is None:
                raise Exception("Invalid JSON response from private GET API")
            if data.get("code") != 0:
                if data.get("code") != 2:
                    app_logger = logging.getLogger("app")
                    app_logger.warning(f"Private GET API Error - Endpoint: {endpoint}, Params: {params}, Response: {data}")
                if data.get("code") == 2:
                    return None  # Treat "System error" as no data (no positions)
                raise Exception(f"API Error {data.get('code')}: {data.get('msg')}")
            return data["data"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _post(self, endpoint: str, data: Dict[str, Any]) -> Any:
        url = f"{API_URL}{endpoint}"
        payload = json.dumps(data, separators=(",", ":"))
        headers = self.auth.get_headers(body=payload)
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers,
                                     content=payload, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            if result is None:
                raise Exception("Invalid JSON response from private POST API")
            if result.get("code") != 0:
                logging.warning(f"Private POST API Error - Endpoint: {endpoint}, Data: {data}, Response: {result}")
                raise Exception(
                    f"API Error {result.get('code')}: {result.get('msg')}")
            return result["data"]

    async def get_current_price(self, symbol: str) -> float:
        data = await self._public_get("/api/v1/futures/market/tickers", {"symbols": symbol})
        for t in data:
            if t["symbol"] == symbol:
                price = float(t["lastPrice"])
                return price
        raise ValueError(f"Price not found for {symbol}")

    async def _get_margin_balance(self, margin_coin: str) -> Tuple[float, float]:
        """Fetches available and total balance for a given margin coin."""
        app_logger = logging.getLogger("app")
        try:
            data = await self._get("/api/v1/futures/account", {})
            app_logger.info(f"Account data for {margin_coin}: {data}")

            # Handle different response formats
            if isinstance(data, list):
                for item in data:
                    if item.get("asset") == margin_coin or item.get("coin") == margin_coin or item.get("currency") == margin_coin:
                        available = float(item.get("available") or item.get("free") or item.get("availableBalance") or 0.0)
                        total = float(item.get("total") or item.get("balance") or item.get("totalBalance") or 0.0)
                        return available, total
            elif isinstance(data, dict):
                # Check if coin is a key
                if margin_coin in data:
                    coin_data = data[margin_coin]
                    if isinstance(coin_data, dict):
                        available = float(coin_data.get("available") or coin_data.get("free") or coin_data.get("availableBalance") or 0.0)
                        total = float(coin_data.get("total") or coin_data.get("balance") or coin_data.get("totalBalance") or 0.0)
                        return available, total
                    elif isinstance(coin_data, (int, float, str)):
                        # If it's a direct value
                        total = float(coin_data)
                        return total, total

                # Check for direct balance fields in root
                available = float(data.get("availableBalance") or data.get("available") or data.get("free") or 0.0)
                total = float(data.get("totalBalance") or data.get("balance") or data.get("walletBalance") or data.get("equity") or 0.0)

                # If we found any balance, return it
                if total > 0 or available > 0:
                    return available, total

                # Last resort: check if the entire dict represents balance
                for key, value in data.items():
                    if isinstance(value, (int, float)) and value > 0:
                        return float(value), float(value)

            return 0.0, 0.0
        except RetryError as e:
            app_logger.warning(
                f"Failed to fetch {margin_coin} balance after multiple retries: {e}")
        except Exception as e:
            # This will catch other exceptions, including API errors
            app_logger.warning(f"Could not fetch {margin_coin} balance: {e}")
        return 0.0, 0.0

    async def get_account_balance(self, currency: str) -> float:
        """Get total account balance"""
        try:
            _, total = await self._get_margin_balance(currency)
            logging.getLogger("app").info(f"Account balance: ${total:,.2f} {currency}")
            return total
        except Exception as e:
            logging.getLogger("app").warning(f"Could not fetch balance: {e}")
            return 0.0

    async def get_pending_positions(self, symbol: str) -> Optional[Position]:
        try:
            # Try different endpoints for positions
            endpoints = [
                "/api/v1/futures/position/list",
                "/api/v1/futures/position/get_positions",
                "/api/v1/futures/position/get_pending_positions"
            ]

            for endpoint in endpoints:
                try:
                    data = await self._get(endpoint, {"symbol": symbol})
                    # Debug: Log the raw API response
                    logging.info(f"Position API response for {symbol} from {endpoint}: {data}")

                    if not data:
                        continue

                    for pos in data:
                        logging.info(f"Checking position: {pos}")
                        qty = float(pos.get("qty", 0))
                        if pos.get("symbol") == symbol and qty != 0:
                            side = "BUY" if qty > 0 else "SELL"
                            return Position(
                                positionId=pos["positionId"],
                                side=side,
                                size=abs(qty),
                                entry_price=float(pos["avgOpenPrice"]),
                                symbol=symbol,
                                timestamp=int(pos["cTime"]),
                            )
                        else:
                            logging.info(f"Position {pos.get('symbol')} qty {qty} - not matching criteria")
                except Exception as e:
                    logging.warning(f"Failed to fetch from {endpoint}: {e}")
                    continue

            logging.warning(f"No position data returned for {symbol} from any endpoint")
            return None
        except Exception as e:
            logging.warning(f"Failed to fetch positions: {e}")
            logging.warning(f"Exception details: {type(e).__name__}: {e}")
        return None

    async def get_all_positions(self, symbol: str) -> List[Position]:
        """Get all positions for the symbol (should only be one for futures)"""
        try:
            # Try different endpoints for positions
            endpoints = [
                "/api/v1/futures/position/list",
                "/api/v1/futures/position/get_positions",
                "/api/v1/futures/position/get_pending_positions"
            ]

            for endpoint in endpoints:
                try:
                    data = await self._get(endpoint, {"symbol": symbol})
                    if data is None:
                        continue
                    positions = []
                    for pos in data:
                        if float(pos.get("qty", 0)) != 0:
                            side = "BUY" if float(pos["qty"]) > 0 else "SELL"
                            # Handle missing cTime field gracefully
                            timestamp = int(pos.get("cTime", time.time() * 1000))
                            positions.append(Position(
                                positionId=pos["positionId"],
                                side=side,
                                size=abs(float(pos["qty"])),
                                entry_price=float(pos["avgOpenPrice"]),
                                symbol=symbol,
                                timestamp=timestamp,
                            ))
                    if positions:  # If we found positions, return them
                        return positions
                except Exception as e:
                    logging.warning(f"Failed to fetch from {endpoint}: {e}")
                    continue

            return []
        except Exception as e:
            logging.warning(f"Failed to fetch all positions: {e}")
            return []

    async def get_account_summary(self, currency: str, symbol: str) -> Dict[str, Any]:
        """Get complete account summary including balance and positions"""
        balance = await self.get_account_balance(currency)
        positions = await self.get_all_positions(symbol)
        current_price = await self.get_current_price(symbol)
        
        summary = {
            "balance": balance,
            "equity": balance,
            "currency": currency,
            "positions": positions,
            "total_positions": len(positions),
            "total_exposure": 0.0,
            "unrealized_pnl": 0.0,
            "leverage": getattr(self.config, 'LEVERAGE', 1)
        }
        
        if positions:
            for position in positions:
                position_value = position.size * current_price
                summary["total_exposure"] += position_value
                # Calculate PnL for each position
                pnl = (current_price - position.entry_price) * position.size
                if position.side == 'SELL':
                    pnl = -pnl
                position.pnl = pnl
                position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                summary["unrealized_pnl"] += pnl
            
            summary["equity"] = balance + summary["unrealized_pnl"]
            
        return summary

    async def set_leverage(self, symbol: str, leverage: int):
        await self._post("/api/v1/futures/account/change_leverage",
                   {"symbol": symbol, "leverage": leverage, "marginCoin": "USDT"})

    async def set_margin_mode(self, symbol: str, mode: str):
        mode_str = "ISOLATION" if mode.upper() == "ISOLATED" else "CROSS"
        await self._post("/api/v1/futures/account/change_margin_mode",
                   {"symbol": symbol, "marginMode": mode_str, "marginCoin": "USDT"})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def open_position(self, symbol: str, side: str, size: str) -> Dict:
        """Open a new position with risk management"""
        try:
            current_price = await self.get_current_price(symbol)

            # Determine position size
            if isinstance(size, str) and '%' in size:
                size_pct = float(size.strip('%')) / 100
                balance = await self.get_account_balance('USDT')
                position_size = (balance * size_pct) / current_price
            else:
                position_size = float(size)

            logging.info(f"Attempting to open position: {side.upper()} {position_size:.4f} {symbol} @ ${current_price:,.2f}")

            # Position size already calculated by trader with risk management
            # No additional adjustment needed here

            if position_size <= 0:
                logging.error("Position size must be positive")
                raise ValueError("Position size must be positive")

            # Check if in forward testing mode (real data, no execution)
            if self.config.get('FORWARD_TESTING', False):
                logging.info(f"📊 FORWARD TESTING: Would open {side.upper()} {position_size:.4f} {symbol} @ ${current_price:,.2f} (NOT EXECUTED)")
                # Return a mock order response
                mock_order = {
                    "orderId": f"forward_test_{int(time.time())}",
                    "status": "simulated",
                    "symbol": symbol,
                    "side": side.upper(),
                    "qty": str(position_size),
                    "price": str(current_price)
                }
                return mock_order

            # Place the main order
            order_data = {
                "symbol": symbol,
                "qty": str(position_size),
                "side": side.upper(),
                "tradeSide": "OPEN",
                "orderType": "MARKET",
                "marginCoin": "USDT",
            }

            logging.info(f"Placing order: {order_data}")
            order = await self._post("/api/v1/futures/trade/place_order", order_data)

            # Log the trade
            trade = {
                'symbol': symbol,
                'side': side.upper(),
                'size': position_size,
                'price': current_price,
                'timestamp': int(time.time() * 1000),
                'type': 'OPEN',
                'pnl': 0,
                'balance': await self.get_account_balance('USDT')
            }

            if self.risk_manager:
                self.risk_manager.update_trade_history(trade)

            logging.info(
                f"✅ Opened {side.upper()} {position_size:.4f} {symbol} @ ${current_price:,.2f}")
            return order

        except Exception as e:
            logging.error(f"❌ Error opening position: {type(e).__name__}: {e}")
            logging.error(traceback.format_exc())
            raise

    async def close_position(self, position: Position, reason: str = "manual") -> Dict:
        """Close an open position with PnL tracking"""
        try:
            side = 'SELL' if position.side == 'BUY' else 'BUY'
            current_price = await self.get_current_price(position.symbol)

            # Check if in forward testing mode (real data, no execution)
            if self.config.get('FORWARD_TESTING', False):
                # Calculate PnL before fees
                pnl = (current_price - position.entry_price) * position.size
                if position.side == 'SELL':
                    pnl = -pnl

                # Deduct taker fee from PnL
                notional_value = position.size * current_price
                taker_fee = self.config.get('TAKER_FEE', 0.0006)
                fee_cost = notional_value * taker_fee
                pnl -= fee_cost

                logging.info(f"📊 FORWARD TESTING: Would close {position.side} {position.size:.4f} {position.symbol} @ "
                           f"${current_price:,.2f} | PnL: ${pnl:,.2f} (fee: ${fee_cost:.2f}) ({reason}) - NOT EXECUTED")

                # Return a mock response
                mock_order = {
                    "orderId": f"forward_test_close_{int(time.time())}",
                    "status": "simulated",
                    "symbol": position.symbol,
                    "side": side,
                    "qty": str(position.size),
                    "price": str(current_price)
                }
                return {'order': mock_order, 'pnl': pnl}

            # Place the order
            order = await self._post("/api/v1/futures/trade/place_order", {
                "symbol": position.symbol,
                "qty": str(position.size),
                "side": side,
                "tradeSide": "CLOSE",
                "orderType": "MARKET",
                "marginCoin": "USDT",
            })

            # Calculate PnL before fees
            pnl = (current_price - position.entry_price) * position.size
            if position.side == 'SELL':
                pnl = -pnl

            # Deduct taker fee from PnL
            notional_value = position.size * current_price
            taker_fee = self.config.get('TAKER_FEE', 0.0006)
            fee_cost = notional_value * taker_fee
            pnl -= fee_cost

            # Log the trade
            trade = {
                'symbol': position.symbol,
                'side': f'CLOSE_{position.side}',
                'size': position.size,
                'price': current_price,
                'entry_price': position.entry_price,
                'timestamp': int(time.time() * 1000),
                'type': 'CLOSE',
                'pnl': pnl,
                'balance': await self.get_account_balance('USDT') + pnl,
                'reason': reason
            }

            if self.risk_manager:
                self.risk_manager.update_trade_history(trade)

            logging.info(f"✅ Closed {position.side} {position.size:.4f} {position.symbol} @ "
                       f"${current_price:,.2f} | PnL: ${pnl:,.2f} ({reason})")
            return {'order': order, 'pnl': pnl}

        except Exception as e:
            logging.error(f"❌ Error closing position: {e}")
            raise

    async def monitor_positions(self):
        """Monitor open positions - simplified AI-driven approach"""
        while True:
            try:
                positions = await self.get_all_positions(self.symbol)
                for position in positions:
                    # No max hold time check - positions held until AI decides to close or SL/TP triggers
                    pass

                # Sleep before next check
                await asyncio.sleep(300)  # Check every 5 minutes (reduced frequency)

            except Exception as e:
                logging.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(300)  # Wait before retry
    
    async def _create_conditional_orders(self, symbol: str, position_size: float, side: str, entry_price: float,
                                        sl_pct: Optional[float], tp_pct: Optional[float]) -> Dict[str, Any]:
        """Create conditional stop loss and take profit orders for a position"""
        results = {
            "sl_order_id": None,
            "tp_order_id": None,
            "sl_success": False,
            "tp_success": False
        }

        try:
            # Validate SL/TP prices are reasonable
            current_price = await self.get_current_price(symbol)
            price_tolerance = 0.001  # 0.1% minimum distance

            # Create Stop Loss order
            if sl_pct is not None and sl_pct > 0:
                sl_price = entry_price * (1 - sl_pct/100) if side.lower() == 'buy' else entry_price * (1 + sl_pct/100)

                # Validate SL price is not too close to entry or current price
                sl_distance = abs(sl_price - entry_price) / entry_price
                if sl_distance < price_tolerance:
                    logging.warning(f"SL price {sl_price:.2f} too close to entry {entry_price:.2f}, skipping SL order")
                else:
                    try:
                        # Try STOP_MARKET first (market order when triggered)
                        sl_order_data = {
                            "symbol": symbol,
                            "qty": str(position_size),
                            "side": 'SELL' if side.upper() == 'BUY' else 'BUY',
                            "tradeSide": "CLOSE",
                            "orderType": "STOP_MARKET",
                            "stopPrice": str(sl_price),
                            "marginCoin": "USDT",
                            "reduceOnly": True
                        }

                        logging.info(f"Creating stop loss order: {sl_order_data}")
                        sl_result = await self._post("/api/v1/futures/trade/place_order", sl_order_data)

                        if sl_result and sl_result.get("orderId"):
                            results["sl_order_id"] = sl_result.get("orderId")
                            results["sl_success"] = True
                            logging.info(f"✅ Conditional stop loss created: {sl_result.get('orderId')}")
                        else:
                            logging.warning(f"SL order creation returned no orderId: {sl_result}")

                    except Exception as sl_cond_err:
                        logging.warning(f"Failed to create conditional stop loss: {sl_cond_err}")
                        try:
                            # Fallback to STOP with limit price
                            sl_order_data["orderType"] = "STOP"
                            sl_order_data["price"] = str(sl_price)
                            sl_result = await self._post("/api/v1/futures/trade/place_order", sl_order_data)
                            if sl_result and sl_result.get("orderId"):
                                results["sl_order_id"] = sl_result.get("orderId")
                                results["sl_success"] = True
                                logging.info(f"✅ Fallback stop loss created: {sl_result.get('orderId')}")
                        except Exception as fallback_err:
                            logging.error(f"❌ Both stop loss attempts failed: {fallback_err}")

            # Create Take Profit order
            if tp_pct is not None and tp_pct > 0:
                # Fee-adjusted TP levels: account for taker fee that will be charged on TP execution
                taker_fee = self.config.get('TAKER_FEE', 0.0006)  # Default 0.06%
                if side.lower() == 'buy':
                    # For long positions: TP price needs to be higher to account for exit fees
                    tp_price = entry_price * (1 + tp_pct/100) / (1 - taker_fee)
                else:
                    # For short positions: TP price needs to be lower to account for exit fees
                    tp_price = entry_price * (1 - tp_pct/100) / (1 + taker_fee)

                # Validate TP price is not too close to entry
                tp_distance = abs(tp_price - entry_price) / entry_price
                if tp_distance < price_tolerance:
                    logging.warning(f"TP price {tp_price:.2f} too close to entry {entry_price:.2f}, skipping TP order")
                else:
                    try:
                        # Try TAKE_PROFIT_MARKET first (market order when triggered)
                        tp_order_data = {
                            "symbol": symbol,
                            "qty": str(position_size),
                            "side": 'SELL' if side.upper() == 'BUY' else 'BUY',
                            "tradeSide": "CLOSE",
                            "orderType": "TAKE_PROFIT_MARKET",
                            "stopPrice": str(tp_price),
                            "marginCoin": "USDT",
                            "reduceOnly": True
                        }

                        logging.info(f"Creating take profit order: {tp_order_data}")
                        tp_result = await self._post("/api/v1/futures/trade/place_order", tp_order_data)

                        if tp_result and tp_result.get("orderId"):
                            results["tp_order_id"] = tp_result.get("orderId")
                            results["tp_success"] = True
                            logging.info(f"✅ Conditional take profit created: {tp_result.get('orderId')}")
                        else:
                            logging.warning(f"TP order creation returned no orderId: {tp_result}")

                    except Exception as tp_cond_err:
                        logging.warning(f"Failed to create conditional take profit: {tp_cond_err}")
                        try:
                            # Fallback to TAKE_PROFIT with limit price
                            tp_order_data["orderType"] = "TAKE_PROFIT"
                            tp_order_data["price"] = str(tp_price)
                            tp_result = await self._post("/api/v1/futures/trade/place_order", tp_order_data)
                            if tp_result and tp_result.get("orderId"):
                                results["tp_order_id"] = tp_result.get("orderId")
                                results["tp_success"] = True
                                logging.info(f"✅ Fallback take profit created: {tp_result.get('orderId')}")
                        except Exception as fallback_err:
                            logging.error(f"❌ Both take profit attempts failed: {fallback_err}")

        except Exception as e:
            logging.error(f"❌ Error creating conditional orders: {e}")
            import traceback
            logging.error(f"Conditional order error details: {traceback.format_exc()}")

        return results

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all open orders for a symbol"""
        try:
            data = await self._get("/api/v1/futures/trade/get_open_orders", {"symbol": symbol})
            return data if data else []
        except Exception as e:
            logging.warning(f"Failed to fetch open orders for {symbol}: {e}")
            return []

    async def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific order"""
        try:
            data = await self._get("/api/v1/futures/trade/get_order_details", {
                "symbol": symbol,
                "orderId": order_id,
                "marginCoin": "USDT"
            })

            if data:
                # Normalize status to common format
                status_mapping = {
                    'filled': 'filled',
                    'partial_filled': 'partial_filled',
                    'pending': 'pending',
                    'cancelled': 'canceled',
                    'rejected': 'rejected',
                    'expired': 'expired'
                }

                raw_status = data.get('status', '').lower()
                normalized_status = status_mapping.get(raw_status, raw_status)

                return {
                    'orderId': data.get('orderId'),
                    'status': normalized_status,
                    'symbol': data.get('symbol'),
                    'side': data.get('side'),
                    'qty': float(data.get('qty', 0)),
                    'filledQty': float(data.get('filledQty', 0)),
                    'price': float(data.get('price', 0)),
                    'avgPrice': float(data.get('avgPrice', 0)),
                    'type': data.get('orderType'),
                    'timestamp': data.get('cTime')
                }

            return None
        except Exception as e:
            logging.warning(f"Failed to get order status for {order_id}: {e}")
            return None

    async def check_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Check order status with validation and detailed error handling"""
        try:
            order_status = await self.get_order_status(symbol, order_id)

            if not order_status:
                return {
                    'valid': False,
                    'status': 'not_found',
                    'error': f'Order {order_id} not found',
                    'order_data': None
                }

            # Validate order data consistency
            validation_errors = []

            # Check required fields
            required_fields = ['orderId', 'status', 'symbol', 'side', 'qty']
            for field in required_fields:
                if field not in order_status or order_status[field] is None:
                    validation_errors.append(f"Missing required field: {field}")

            # Validate quantities
            qty = order_status.get('qty', 0)
            filled_qty = order_status.get('filledQty', 0)

            if qty < 0:
                validation_errors.append("Order quantity cannot be negative")
            if filled_qty < 0:
                validation_errors.append("Filled quantity cannot be negative")
            if filled_qty > qty:
                validation_errors.append(f"Filled quantity ({filled_qty}) exceeds order quantity ({qty})")

            # Validate prices
            price = order_status.get('price', 0)
            avg_price = order_status.get('avgPrice', 0)

            if price < 0:
                validation_errors.append("Order price cannot be negative")
            if avg_price < 0:
                validation_errors.append("Average price cannot be negative")

            # Validate status transitions
            valid_statuses = ['pending', 'partial_filled', 'filled', 'canceled', 'rejected', 'expired']
            if order_status.get('status') not in valid_statuses:
                validation_errors.append(f"Invalid status: {order_status.get('status')}")

            # Check for logical inconsistencies
            status = order_status.get('status')
            if status == 'filled' and filled_qty == 0:
                validation_errors.append("Order marked as filled but filled quantity is zero")
            elif status == 'filled' and filled_qty < qty:
                validation_errors.append("Order marked as filled but not fully filled")
            elif status in ['canceled', 'rejected', 'expired'] and filled_qty > 0:
                validation_errors.append(f"Order {status} but has filled quantity")

            return {
                'valid': len(validation_errors) == 0,
                'status': order_status.get('status'),
                'validation_errors': validation_errors,
                'order_data': order_status
            }

        except Exception as e:
            logging.error(f"Error checking order status for {order_id}: {e}")
            return {
                'valid': False,
                'status': 'error',
                'error': str(e),
                'order_data': None
            }

    async def validate_position_data(self, symbol: str) -> Dict[str, Any]:
        """Validate position data consistency and integrity"""
        try:
            # Get position data from multiple sources for cross-validation
            pending_position = await self.get_pending_positions(symbol)
            all_positions = await self.get_all_positions(symbol)

            validation_result = {
                'valid': True,
                'validation_errors': [],
                'warnings': [],
                'position_count': len(all_positions),
                'pending_position': pending_position,
                'all_positions': all_positions
            }

            # Validate position count consistency
            if len(all_positions) > 1:
                validation_result['warnings'].append(f"Multiple positions found for {symbol}: {len(all_positions)} positions")

            # Validate pending position vs all positions consistency
            if pending_position and all_positions:
                pending_in_all = any(pos.positionId == pending_position.positionId for pos in all_positions)
                if not pending_in_all:
                    validation_result['validation_errors'].append("Pending position not found in all positions list")
                    validation_result['valid'] = False

            # Validate each position's data integrity
            for i, position in enumerate(all_positions):
                position_errors = []
                position_warnings = []

                # Check required fields
                if not hasattr(position, 'positionId') or not position.positionId:
                    position_errors.append(f"Position {i}: Missing positionId")
                if not hasattr(position, 'symbol') or position.symbol != symbol:
                    position_errors.append(f"Position {i}: Symbol mismatch (expected {symbol}, got {position.symbol})")
                if not hasattr(position, 'side') or position.side not in ['BUY', 'SELL']:
                    position_errors.append(f"Position {i}: Invalid side '{position.side}'")
                if not hasattr(position, 'size') or position.size <= 0:
                    position_errors.append(f"Position {i}: Invalid size {position.size}")
                if not hasattr(position, 'entry_price') or position.entry_price <= 0:
                    position_errors.append(f"Position {i}: Invalid entry price {position.entry_price}")
                if not hasattr(position, 'timestamp') or position.timestamp <= 0:
                    position_errors.append(f"Position {i}: Invalid timestamp {position.timestamp}")

                # Validate position size is reasonable
                try:
                    current_price = await self.get_current_price(symbol)
                    position_value = position.size * current_price
                    if position_value < 1:  # Less than $1 position value
                        position_warnings.append(f"Position {i}: Very small position value (${position_value:.2f})")
                    elif position_value > 1000000:  # More than $1M position value
                        position_warnings.append(f"Position {i}: Very large position value (${position_value:.2f})")
                except Exception as e:
                    position_warnings.append(f"Position {i}: Could not validate position value: {e}")

                # Validate timestamp is reasonable (not in future, not too old)
                current_time = int(time.time() * 1000)
                if position.timestamp > current_time + 60000:  # More than 1 minute in future
                    position_errors.append(f"Position {i}: Timestamp is in the future")
                elif position.timestamp < current_time - (365 * 24 * 60 * 60 * 1000):  # More than 1 year old
                    position_warnings.append(f"Position {i}: Very old timestamp")

                # Validate entry price is reasonable compared to current price
                try:
                    current_price = await self.get_current_price(symbol)
                    price_ratio = position.entry_price / current_price
                    if price_ratio < 0.1 or price_ratio > 10:  # More than 10x deviation
                        position_warnings.append(f"Position {i}: Entry price significantly different from current price (ratio: {price_ratio:.2f})")
                except Exception as e:
                    position_warnings.append(f"Position {i}: Could not validate entry price: {e}")

                # Add position-specific errors and warnings
                if position_errors:
                    validation_result['validation_errors'].extend(position_errors)
                    validation_result['valid'] = False
                if position_warnings:
                    validation_result['warnings'].extend(position_warnings)

            # Cross-validate with account balance if position exists
            if all_positions:
                try:
                    account_balance = await self.get_account_balance('USDT')
                    total_exposure = sum(pos.size * await self.get_current_price(symbol) for pos in all_positions)
                    exposure_ratio = total_exposure / account_balance if account_balance > 0 else 0

                    if exposure_ratio > 10:  # More than 10x leverage implied
                        validation_result['warnings'].append(f"High exposure ratio: {exposure_ratio:.2f}x (balance: ${account_balance:.2f}, exposure: ${total_exposure:.2f})")
                except Exception as e:
                    validation_result['warnings'].append(f"Could not validate exposure ratio: {e}")

            return validation_result

        except Exception as e:
            logging.error(f"Error validating position data for {symbol}: {e}")
            return {
                'valid': False,
                'validation_errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'position_count': 0,
                'pending_position': None,
                'all_positions': []
            }

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            result = await self._post("/api/v1/futures/trade/cancel_order", {
                "symbol": symbol,
                "orderId": order_id,
                "marginCoin": "USDT"
            })
            return result is not None
        except Exception as e:
            logging.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    async def start_monitoring(self):
        """Start the position monitoring task"""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self.monitor_positions())
    
    async def close(self):
        """Cleanup resources"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
    
    async def flash_close_position(self, symbol: str) -> None:
        """Instantly close any open position for the symbol"""
        position = await self.get_pending_positions(symbol)
        if position:
            # Calculate PnL before closing (fees will be deducted in close_position)
            current_price = await self.get_current_price(symbol)
            pnl = (current_price - position.entry_price) * position.size
            if position.side == 'SELL':
                pnl = -pnl

            # Close the position (this will deduct fees)
            result = await self.close_position(position, reason="Emergency close")
            final_pnl = result.get('pnl', pnl)

            # Calculate percentage after fees
            pnl_pct = (final_pnl / (position.entry_price * position.size)) * 100 if (position.entry_price * position.size) != 0 else 0

            # Log concise closing details
            logging.info(
                f"🔒 Closed {position.side} {position.size:.4f} {symbol} | PnL: ${final_pnl:+.2f} ({pnl_pct:+.2f}%)")
            return

        logging.info(f"No open position found for {symbol} to flash close.")

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 15) -> list:
        data = await self._public_get(
            "/api/v1/futures/market/kline", {"symbol": symbol, "interval": timeframe, "limit": limit})
        if data is None:
            return []
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

    def _get_supported_interval(self, timeframe_minutes: int) -> str:
        """Map timeframe minutes to supported Bitunix interval"""
        if timeframe_minutes <= 1:
            return "1m"
        elif timeframe_minutes <= 5:
            return "5m"
        elif timeframe_minutes <= 15:
            return "15m"
        elif timeframe_minutes <= 30:
            return "30m"
        elif timeframe_minutes <= 60:
            return "1h"
        elif timeframe_minutes <= 240:
            return "4h"
        elif timeframe_minutes <= 1440:
            return "1d"
        else:
            return "1d"  # fallback

    async def get_ohlcv(self, symbol: str, timeframe: Any, limit: int) -> list:
        tf_minutes = int(timeframe)
        tf_str = self._get_supported_interval(tf_minutes)
        return await self.fetch_ohlcv(symbol, tf_str, limit)

    async def get_ticker(self, symbol: str) -> float:
        return await self.get_current_price(symbol)

    async def set_position_tp_sl(self, symbol: str, position_id: str, sl_price: float, tp_price: float):
        """Set stop loss and take profit on an existing position"""
        data = {
            "symbol": symbol,
            "positionId": position_id,
            "slPrice": str(sl_price),
            "tpPrice": str(tp_price)
        }
        return await self._post("/api/v1/futures/trade/place_position_tp_sl_order", data)

    async def place_order(self, symbol: str, side: str, type: str, quantity: float, leverage: Optional[int] = None) -> Dict[str, Any]:
        # Check if in forward testing mode (real data, no execution)
        if self.config.get('FORWARD_TESTING', False):
            logging.info(f"📊 FORWARD TESTING: Would place {type} order {side.upper()} {quantity:.6f} {symbol} (NOT EXECUTED)")
            return {"status": "simulated", "orderId": f"forward_test_{int(time.time())}", "pnl": 0.0}

        if leverage is not None:
            await self.set_leverage(symbol, leverage)
        position = await self.get_pending_positions(symbol)
        trade_side = "CLOSE" if position else "OPEN"
        order_data = {
            "symbol": symbol,
            "qty": f"{quantity:.6f}",
            "side": side.upper(),
            "tradeSide": trade_side,
            "orderType": type,
            "marginCoin": "USDT",
        }
        order = await self._post("/api/v1/futures/trade/place_order", order_data)
        return {"status": "filled", "orderId": order.get("orderId", "unknown") if order else "failed", "pnl": 0.0}
