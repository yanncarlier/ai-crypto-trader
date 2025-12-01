# core/decision_engine.py
def get_action(interpretation: str, current_position: str | None) -> str:
    interp = interpretation.upper()
    pos = current_position.lower() if current_position else None
    match (interp, pos):
        case ("BULLISH", None): return "OPEN_LONG"
        case ("BULLISH", "sell"): return "REVERSE_TO_LONG"
        case ("BULLISH", "buy"): return "HOLD"
        case ("BEARISH", None): return "OPEN_SHORT"
        case ("BEARISH", "buy"): return "REVERSE_TO_SHORT"
        case ("BEARISH", "sell"): return "HOLD"
        case ("NEUTRAL", _): return "CLOSE" if pos else "STAY_FLAT"
        case _: return "HOLD"
