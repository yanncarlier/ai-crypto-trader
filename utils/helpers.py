import logging


def open_position(exchange, symbol, side, size, sl_pct):
    try:
        exchange.open_position(symbol, side, size, sl_pct)
        logging.info(f"Position opened: {side.upper()} {size}")
    except Exception as e:
        logging.error(f"Failed to open position: {e}")
