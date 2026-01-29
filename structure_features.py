SWING_HIGH = "SWING_HIGH"
SWING_LOW = "SWING_LOW"
SWING_NONE = "NONE"
LEG_IMPULSE = "IMPULSE"
LEG_CORRECTION = "CORRECTION"


def detect_swings(prices, lookback: int = 2, lookforward: int = 2):
    n = len(prices)
    tags = [SWING_NONE for _ in range(n)]
    if n == 0:
        return tags
    for i in range(n):
        if i < lookback or i + lookforward >= n:
            continue
        p = prices[i]
        left = prices[i - lookback : i]
        right = prices[i + 1 : i + 1 + lookforward]
        if not left or not right:
            continue
        if p >= max(left) and p > max(right):
            tags[i] = SWING_HIGH
        elif p <= min(left) and p < min(right):
            tags[i] = SWING_LOW
    return tags


def compute_structure_features(prices, lookback: int = 2, lookforward: int = 2):
    if not prices:
        return {
            "swing_tag": SWING_NONE,
            "current_leg_type": LEG_CORRECTION,
            "last_bos_direction": "NONE",
            "last_choch_direction": "NONE",
        }

    swings = detect_swings(prices, lookback, lookforward)
    last_swing_high = None
    last_swing_low = None
    bos_direction = "NONE"
    choch_direction = "NONE"

    for idx, price in enumerate(prices):
        tag = swings[idx]
        if tag == SWING_HIGH:
            last_swing_high = price
        elif tag == SWING_LOW:
            last_swing_low = price

        if last_swing_high is not None and price > last_swing_high:
            if bos_direction == "DOWN":
                choch_direction = "UP"
            bos_direction = "UP"
        if last_swing_low is not None and price < last_swing_low:
            if bos_direction == "UP":
                choch_direction = "DOWN"
            bos_direction = "DOWN"

    last_price = prices[-1]
    if last_swing_high is not None and last_price > last_swing_high:
        leg_type = LEG_IMPULSE
    elif last_swing_low is not None and last_price < last_swing_low:
        leg_type = LEG_IMPULSE
    else:
        leg_type = LEG_CORRECTION

    return {
        "swing_tag": swings[-1],
        "current_leg_type": leg_type,
        "last_bos_direction": bos_direction,
        "last_choch_direction": choch_direction,
    }
