# core/decision_engine.py
def get_action(interpretation: str, current_position: str | None) -> str:
    """
    Determine trading action based on AI interpretation and current position.
    Args:
        interpretation: "Bullish", "Bearish", or "Neutral"
        current_position: "buy", "sell", or None
    Returns:
        Action: "OPEN_LONG", "OPEN_SHORT", "REVERSE_TO_LONG", 
                "REVERSE_TO_SHORT", "CLOSE", "HOLD", or "STAY_FLAT"
    """
    interp = interpretation.upper()
    pos = current_position.lower() if current_position else None
    match (interp, pos):
        case ("BULLISH", None):
            return "OPEN_LONG"
        case ("BULLISH", "sell"):
            return "REVERSE_TO_LONG"
        case ("BULLISH", "buy"):
            return "HOLD"
        case ("BEARISH", None):
            return "OPEN_SHORT"
        case ("BEARISH", "buy"):
            return "REVERSE_TO_SHORT"
        case ("BEARISH", "sell"):
            return "HOLD"
        case ("NEUTRAL", "buy" | "sell"):
            return "CLOSE"
        case ("NEUTRAL", None):
            return "STAY_FLAT"
        case _:
            return "HOLD"
