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
                logging.warning(f"Private GET API Error - Endpoint: {endpoint}, Params: {params}, Response: {data}")
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
        data = await self._public_get("/market/tickers", {"symbols": symbol})
        for t in data:
            if t["symbol"] == symbol:
                price = float(t["lastPrice"])
                return price
        raise ValueError(f"Price not found for {symbol}")

    async def _get_margin_balance(self, margin_coin: str) -> Tuple[float, float]:
        """Fetches available and total balance for a given margin coin."""
        try:
            data = await self._get("/account", {"marginCoin": margin_coin})
            available = float(data.get("available") or 0.0)
            total = float(data.get("total") or 0.0)
            return available, total
        except RetryError as e:
            logging.warning(
                f"Failed to fetch {margin_coin} balance after multiple retries: {e}")
        except Exception as e:
            # This will catch other exceptions, including API errors
            logging.warning(f"Could not fetch {margin_coin} balance: {e}")
        return 0.0, 0.0

    async def get_account_balance(self, currency: str) -> float:
        """Get total account balance in USDT (including other assets converted to USDT)"""
        # Fetch USDT balance
        usdt_balance, total_usdt = await self._get_margin_balance("USDT")
        # Fetch BTC balance
        btc_balance, total_btc = await self._get_margin_balance("BTC")

        # Safely get current BTC price
        try:
            btc_price = await self.get_current_price("BTCUSDT")
        except Exception as e:
            logging.error(f"❌ Failed to fetch BTC price: {e}")
            logging.warning(
                f"💰 Balance: ${usdt_balance:,.2f} USDT (BTC not included)")
            return usdt_balance

        # Calculate total values
        btc_in_usdt = btc_balance * btc_price
        total_btc_in_usdt = total_btc * btc_price
        total_available = usdt_balance + btc_in_usdt
        total_equity = total_usdt + total_btc_in_usdt

        # Log concise balance summary
        if btc_balance > 0:
            logging.info(
                f"💰 Balance: ${total_available:,.2f} (${usdt_balance:,.2f} USDT + {btc_balance:.6f} BTC)")
        else:
            logging.info(f"💰 Balance: ${total_available:,.2f} USDT")

        # Return total available balance in USDT
        return total_available

    async def get_pending_positions(self, symbol: str) -> Optional[Position]:
        try:
            data = await self._get("/position/get_pending_positions",
                             {"symbol": symbol})
            if not data:
                return None
            for pos in data:
                if pos.get("symbol") == symbol and float(pos.get("qty", 0)) != 0:
                    side = "BUY" if float(pos["qty"]) > 0 else "SELL"
                    return Position(
                        positionId=pos["positionId"],
                        side=side,
                        size=abs(float(pos["qty"])),
                        entry_price=float(pos["avgOpenPrice"]),
                        symbol=symbol,
                        timestamp=int(pos["cTime"]),
                    )
        except Exception as e:
            logging.warning(f"Failed to fetch positions: {e}")
        return None

    async def get_all_positions(self, symbol: str) -> List[Position]:
        """Get all positions for the symbol (should only be one for futures)"""
        try:
            data = await self._get("/position/get_pending_positions",
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
                        timestamp=int(pos["cTime"]),
                    ))
            return positions
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
        await self._post("/account/change_leverage",
                   {"symbol": symbol, "leverage": leverage, "marginCoin": "USDT"})

    async def set_margin_mode(self, symbol: str, mode: str):
        mode_str = "ISOLATION" if mode.upper() == "ISOLATED" else "CROSS"
        await self._post("/account/change_margin_mode",
                   {"symbol": symbol, "marginMode": mode_str, "marginCoin": "USDT"})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def open_position(self, symbol: str, side: str, size: str,
                          sl_pct: Optional[float] = None,
                          tp_pct: Optional[float] = None) -> Dict:
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

            # Apply risk management with fallback
            if self.risk_manager:
                try:
                    can_trade, reason = await self.risk_manager.check_risk_limits(symbol)
                    if not can_trade:
                        logging.warning(f"Risk check failed: {reason}")
                        # Allow trade to proceed with smaller size as fallback
                        position_size = min(position_size, (position_size * 0.1))
                except Exception as risk_err:
                    logging.warning(f"Risk check error, proceeding cautiously: {risk_err}")
                    
                # Try to use risk manager for position sizing, but don't fail if it doesn't work
                if self.risk_manager.risk_params.volatility_adjusted:
                    try:
                        adjusted_size, details = await self.risk_manager.calculate_position_size(symbol, current_price, position_size)
                        if adjusted_size > 0:
                            position_size = adjusted_size
                            logging.info(f"Position size adjusted by risk manager: {position_size}")
                    except Exception as sizing_err:
                        logging.warning(f"Failed to adjust position size with volatility, using original size: {sizing_err}")

            if position_size <= 0:
                logging.error("Position size must be positive")
                raise ValueError("Position size must be positive")

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
            order = await self._post("/trade/place_order", order_data)

            # Set stop loss if specified
            if sl_pct is not None and sl_pct > 0:
                try:
                    sl_price = current_price * (1 - sl_pct/100) if side.lower() == 'buy' else current_price * (1 + sl_pct/100)
                    sl_order = await self._post("/trade/place_order", {
                        "symbol": symbol,
                        "qty": str(position_size),
                        "side": 'SELL' if side.upper() == 'BUY' else 'BUY',
                        "tradeSide": "CLOSE",  
                        "orderType": "STOP_MARKET",
                        "stopPrice": str(sl_price),
                        "marginCoin": "USDT",
                    })
                    logging.info(f"✅ Stop loss set at {sl_price:,.2f} (order: {sl_order.get('orderId', 'unknown')})")
                except Exception as sl_err:
                    logging.error(f"❌ Failed to set stop loss: {sl_err}")
                    # Log the full error for debugging
                    import traceback
                    logging.error(f"Stop loss error details: {traceback.format_exc()}")

            # Set take profit if specified
            if tp_pct is not None and tp_pct > 0:
                try:
                    tp_price = current_price * (1 + tp_pct/100) if side.lower() == 'buy' else current_price * (1 - tp_pct/100)
                    tp_order = await self._post("/trade/place_order", {
                        "symbol": symbol,
                        "qty": str(position_size),
                        "side": 'SELL' if side.upper() == 'BUY' else 'BUY',
                        "tradeSide": "CLOSE",  
                        "orderType": "TAKE_PROFIT_MARKET",
                        "stopPrice": str(tp_price),
                        "marginCoin": "USDT",
                    })
                    logging.info(f"✅ Take profit set at {tp_price:,.2f} (order: {tp_order.get('orderId', 'unknown')})")
                except Exception as tp_err:
                    logging.error(f"❌ Failed to set take profit: {tp_err}")
                    # Log the full error for debugging
                    import traceback
                    logging.error(f"Take profit error details: {traceback.format_exc()}")
            
            # Also try to create conditional orders using alternative approach
            if (sl_pct is not None and sl_pct > 0) or (tp_pct is not None and tp_pct > 0):
                await self._create_conditional_orders(symbol, position_size, side, current_price, sl_pct, tp_pct)

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
            
            # Place the order
            order = await self._post("/trade/place_order", {
                "symbol": position.symbol,
                "qty": str(position.size),
                "side": side,
                "tradeSide": "CLOSE",
                "orderType": "MARKET",
                "marginCoin": "USDT",
            })
            
            # Calculate PnL
            pnl = (current_price - position.entry_price) * position.size
            if position.side == 'SELL':
                pnl = -pnl
            
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
            return order
            
        except Exception as e:
            logging.error(f"❌ Error closing position: {e}")
            raise

    async def monitor_positions(self):
        """Monitor open positions and apply risk management rules"""
        while True:
            try:
                positions = await self.get_all_positions(self.symbol)
                for position in positions:
                    # Check if position is held too long
                    position_age = (time.time() * 1000 - position.timestamp) / (1000 * 3600)
                    max_hold_hours = self.risk_manager.risk_params.max_hold_period_hours if self.risk_manager else 24
                    
                    if position_age > max_hold_hours:
                        await self.close_position(position, reason="Max hold time reached")
                        continue
                        
                    # Check if stop loss or take profit is hit
                    current_price = await self.get_current_price(position.symbol)
                    if hasattr(position, 'stop_loss'):
                        if (position.side == 'BUY' and current_price <= position.stop_loss) or \
                           (position.side == 'SELL' and current_price >= position.stop_loss):
                            await self.close_position(position, reason="Stop loss")
                            continue
                            
                    if hasattr(position, 'take_profit'):
                        if (position.side == 'BUY' and current_price >= position.take_profit) or \
                           (position.side == 'SELL' and current_price <= position.take_profit):
                            await self.close_position(position, reason="Take profit")
                            continue
                            
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _create_conditional_orders(self, symbol: str, position_size: float, side: str, entry_price: float, 
                                        sl_pct: Optional[float], tp_pct: Optional[float]) -> Dict[str, Any]:
        """Create conditional stop loss and take profit orders for a position"""
        results = {}
        
        try:
            # Create Stop Loss order
            if sl_pct is not None and sl_pct > 0:
                try:
                    sl_price = entry_price * (1 - sl_pct/100) if side.lower() == 'buy' else entry_price * (1 + sl_pct/100)
                    
                    # Try with different parameter structure first
                    sl_order_data = {
                        "symbol": symbol,
                        "qty": str(position_size),
                        "side": 'SELL' if side.upper() == 'BUY' else 'BUY',
                        "tradeSide": "CLOSE",
                        "orderType": "STOP",
                        "price": str(sl_price),
                        "stopPrice": str(sl_price),
                        "marginCoin": "USDT",
                        "reduceOnly": True
                    }
                    
                    logging.info(f"Creating stop loss order: {sl_order_data}")
                    sl_result = await self._post("/trade/place_order", sl_order_data)
                    
                    if sl_result and sl_result.get("orderId"):
                        results["stop_loss"] = sl_result
                        logging.info(f"✅ Conditional stop loss created: {sl_result.get('orderId')}")
                    
                except Exception as sl_cond_err:
                    logging.warning(f"Failed to create conditional stop loss: {sl_cond_err}")
                    try:
                        # Fallback to standard approach
                        sl_order_data["orderType"] = "STOP_MARKET"
                        sl_result = await self._post("/trade/place_order", sl_order_data)
                        if sl_result and sl_result.get("orderId"):
                            results["stop_loss"] = sl_result
                            logging.info(f"✅ Fallback stop loss created: {sl_result.get('orderId')}")
                    except Exception as fallback_err:
                        logging.error(f"❌ Both stop loss attempts failed: {fallback_err}")
                
            # Create Take Profit order
            if tp_pct is not None and tp_pct > 0:
                try:
                    tp_price = entry_price * (1 + tp_pct/100) if side.lower() == 'buy' else entry_price * (1 - tp_pct/100)
                    
                    # Try with different parameter structure first
                    tp_order_data = {
                        "symbol": symbol,
                        "qty": str(position_size),
                        "side": 'SELL' if side.upper() == 'BUY' else 'BUY',
                        "tradeSide": "CLOSE",
                        "orderType": "TAKE_PROFIT",
                        "price": str(tp_price),
                        "stopPrice": str(tp_price),
                        "marginCoin": "USDT",
                        "reduceOnly": True
                    }
                    
                    logging.info(f"Creating take profit order: {tp_order_data}")
                    tp_result = await self._post("/trade/place_order", tp_order_data)
                    
                    if tp_result and tp_result.get("orderId"):
                        results["take_profit"] = tp_result
                        logging.info(f"✅ Conditional take profit created: {tp_result.get('orderId')}")
                        
                except Exception as tp_cond_err:
                    logging.warning(f"Failed to create conditional take profit: {tp_cond_err}")
                    try:
                        # Fallback to standard approach
                        tp_order_data["orderType"] = "TAKE_PROFIT_MARKET"
                        tp_result = await self._post("/trade/place_order", tp_order_data)
                        if tp_result and tp_result.get("orderId"):
                            results["take_profit"] = tp_result
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
            data = await self._get("/trade/get_open_orders", {"symbol": symbol})
            return data if data else []
        except Exception as e:
            logging.warning(f"Failed to fetch open orders for {symbol}: {e}")
            return []

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            result = await self._post("/trade/cancel_order", {
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
            # Calculate PnL before closing
            current_price = await self.get_current_price(symbol)
            pnl = (current_price - position.entry_price) * position.size
            if position.side == 'SELL':
                pnl = -pnl
            pnl_pct = (pnl / (position.entry_price * position.size)) * 100 if (position.entry_price * position.size) != 0 else 0

            # Close the position
            await self.close_position(position, reason="Emergency close")

            # Log concise closing details
            logging.info(
                f"🔒 Closed {position.side} {position.size:.4f} {symbol} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            return
        
        logging.info(f"No open position found for {symbol} to flash close.")

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 15) -> list:
        data = await self._public_get(
            "/market/kline", {"symbol": symbol, "interval": timeframe, "limit": limit})
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

    async def get_ohlcv(self, symbol: str, timeframe: Any, limit: int) -> list:
        tf_str = f"{int(timeframe)}m"
        return await self.fetch_ohlcv(symbol, tf_str, limit)

    async def get_ticker(self, symbol: str) -> float:
        return await self.get_current_price(symbol)

    async def place_order(self, symbol: str, side: str, type: str, quantity: float, leverage: Optional[int] = None) -> Dict[str, Any]:
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
        order = await self._post("/trade/place_order", order_data)
        return {"status": "filled", "orderId": order.get("orderId", "unknown") if order else "failed", "pnl": 0.0}
