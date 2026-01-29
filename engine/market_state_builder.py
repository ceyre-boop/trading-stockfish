"""
Market State Builder v4.0‑C
Injects liquidity block into MarketState.
"""

from typing import Dict, List, Optional

from engine.amd_features import AMDFeatures
from engine.liquidity_features import LiquidityFeatures
from engine.order_book_model import OrderBookModel
from engine.order_flow_features import OrderFlowFeatures
from engine.trend_structure import compute_trend_structure
from engine.volatility_utils import compute_atr
from ict_smc_features import compute_ict_smc_features
from levels_features import compute_level_features
from liquidity_depth_features import compute_depth_features
from liquidity_primitives import compute_liquidity_primitives
from momentum_features import compute_momentum_features
from orderflow_features import compute_orderflow_features
from pattern_stats import compute_pattern_probabilities
from session_regime import SessionRegime, compute_session_regime
from structure_features import compute_structure_features
from trend_indicator_features import compute_trend_indicator_features


class LiquidityState:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.__dict__


def build_market_state(
    symbol: str,
    order_book_events: Optional[List[Dict]] = None,
    use_microstructure_realism: bool = True,
    timestamp: Optional[float] = None,
) -> Dict:
    # Volatility and regime imports
    from engine.regime_engine import RegimeEngine
    from engine.volatility_features import VolatilityFeatures

    volatility = VolatilityFeatures(
        window=20, use_microstructure_realism=use_microstructure_realism
    )
    regime_engine = RegimeEngine(window=10)

    order_book = OrderBookModel(depth=5)
    order_flow = OrderFlowFeatures(
        lookback=10, use_microstructure_realism=use_microstructure_realism
    )
    liquidity = LiquidityFeatures(
        window=5, use_microstructure_realism=use_microstructure_realism
    )
    amd = AMDFeatures(window=50)

    mid_prices: List[float] = []
    price_volumes: List[float] = []
    volumes: List[float] = []
    timestamps_series: List[float] = []
    session_series: List[str] = []
    aggressive_buy_series: List[float] = []
    aggressive_sell_series: List[float] = []

    for event in order_book_events or []:
        order_book.update_from_event(event)
        ob_snapshot = order_book.get_depth_snapshot()
        order_flow.update(ob_snapshot, event)
        best_bid, best_ask = order_book.get_best_bid_ask()
        ts_evt = float(event.get("timestamp", len(mid_prices)))
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
            mid_prices.append(mid)
            timestamps_series.append(ts_evt)
            price_volumes.append(float(event.get("size", 1.0)))
            session_series.append(compute_session_regime(ts_evt).value)
            aggressive_buy_series.append(0.0)
            aggressive_sell_series.append(0.0)
        if event.get("type") == "trade":
            volumes.append(float(event.get("size", 1.0)))
            if mid_prices:
                side = event.get("aggressor") or event.get("side")
                size = float(event.get("size", 1.0) or 0.0)
                if side == "buy":
                    aggressive_buy_series[-1] += size
                elif side == "sell":
                    aggressive_sell_series[-1] += size
        if mid_prices:
            # Keep volatility state in sync with processed events
            atr_value = compute_atr(mid_prices, window=volatility.atr_window)
            volatility.compute(
                mid_prices[-1],
                candle_data={"atr": atr_value, "atr_14": atr_value},
            )

    ob_snapshot = order_book.get_depth_snapshot()
    depth_features = compute_depth_features(ob_snapshot)
    order_flow_inputs = order_flow.get_liquidity_inputs()
    liquidity_features = liquidity.compute(
        ob_snapshot, order_flow_inputs=order_flow_inputs
    )
    liquidity_state = LiquidityState(**liquidity_features)

    # Compute volatility features using collected mid prices
    if mid_prices:
        atr_value = compute_atr(mid_prices, window=volatility.atr_window)
        volatility_state = volatility.compute(
            mid_prices[-1],
            candle_data={"atr": atr_value, "atr_14": atr_value},
        )
    else:
        volatility_state = volatility._neutral()

    trend_struct = compute_trend_structure(
        mid_prices,
        window=20,
        volatility_state=volatility_state,
    )

    amd_state = amd.compute(
        price_series=mid_prices,
        volume_series=volumes,
        liquidity_state=liquidity_state.to_dict(),
    )

    # Dummy macro_news_state for now (can be extended)
    macro_news_state = {"hawkishness": 0.0, "risk_sentiment": 0.0}
    ml_state = {
        "recent_returns": volatility_state.get("recent_returns", []),
        "volatility_state": volatility_state,
    }
    session_label = (
        session_series[-1]
        if session_series
        else compute_session_regime(timestamp or 0.0).value
    )
    regime_state = regime_engine.compute(
        volatility_state,
        liquidity_state.to_dict(),
        macro_news_state,
        ml_state=ml_state,
        amd_state=amd_state,
        price_series=mid_prices,
        session_regime=session_label,
    )

    momentum_features = compute_momentum_features(mid_prices)
    structure_features = compute_structure_features(mid_prices)
    level_features = compute_level_features(
        mid_prices,
        timestamps=timestamps_series,
        volumes=price_volumes,
        session_labels=session_series,
        default_session=session_label,
    )
    liquidity_primitives = compute_liquidity_primitives(mid_prices)
    ict_smc_features = compute_ict_smc_features(
        mid_prices, timestamps=timestamps_series
    )
    orderflow_metrics = compute_orderflow_features(
        mid_prices,
        aggressive_buy=aggressive_buy_series,
        aggressive_sell=aggressive_sell_series,
    )
    pattern_probs = compute_pattern_probabilities([])
    trend_indicator = compute_trend_indicator_features(mid_prices)

    # Governance metadata (dummy for now, to be updated by engine)
    governance_metadata = {
        "recent_vetoes": [],
        "drawdown": 0.0,
        "trade_frequency": 0,
        "regime_transition": regime_state.get("regime_transition", False),
    }

    state = {
        "symbol": symbol,
        "order_book": ob_snapshot,
        "order_flow": order_flow.get_features(),
        "liquidity_state": liquidity_state.to_dict(),
        "volatility_state": volatility_state,
        "session_regime": session_label,
        "swing_high": trend_struct["swing_high"],
        "swing_low": trend_struct["swing_low"],
        "swing_structure": trend_struct["swing_structure"],
        "trend_direction": trend_struct["trend_direction"],
        "trend_strength": trend_indicator["trend_strength"],
        "trend_strength_state": trend_indicator["trend_strength_state"],
        "swing_tag": structure_features["swing_tag"],
        "current_leg_type": structure_features["current_leg_type"],
        "last_bos_direction": structure_features["last_bos_direction"],
        "last_choch_direction": structure_features["last_choch_direction"],
        "regime_state": regime_state,
        "amd_state": amd_state,
        "bid_depth": depth_features["bid_depth"],
        "ask_depth": depth_features["ask_depth"],
        "depth_imbalance": depth_features["depth_imbalance"],
        "top_of_book_spread": depth_features["top_of_book_spread"],
        "session_high": level_features["session_high"],
        "session_low": level_features["session_low"],
        "day_high": level_features["day_high"],
        "day_low": level_features["day_low"],
        "previous_day_high": level_features["previous_day_high"],
        "previous_day_low": level_features["previous_day_low"],
        "previous_day_close": level_features["previous_day_close"],
        "vwap_price": level_features["vwap_price"],
        "distance_from_vwap": level_features["distance_from_vwap"],
        "has_equal_highs": liquidity_primitives["has_equal_highs"],
        "has_equal_lows": liquidity_primitives["has_equal_lows"],
        "bsl_zone_price": liquidity_primitives["bsl_zone_price"],
        "ssl_zone_price": liquidity_primitives["ssl_zone_price"],
        "nearest_bsl_pool_above": liquidity_primitives["nearest_bsl_pool_above"],
        "nearest_ssl_pool_below": liquidity_primitives["nearest_ssl_pool_below"],
        "has_liquidity_void": liquidity_primitives["has_liquidity_void"],
        "void_upper": liquidity_primitives["void_upper"],
        "void_lower": liquidity_primitives["void_lower"],
        "stop_cluster_above": liquidity_primitives["stop_cluster_above"],
        "stop_cluster_below": liquidity_primitives["stop_cluster_below"],
        "last_sweep_direction": liquidity_primitives["last_sweep_direction"],
        "swept_bsl": liquidity_primitives["swept_bsl"],
        "swept_ssl": liquidity_primitives["swept_ssl"],
        "current_bullish_ob_low": ict_smc_features["current_bullish_ob_low"],
        "current_bullish_ob_high": ict_smc_features["current_bullish_ob_high"],
        "current_bearish_ob_low": ict_smc_features["current_bearish_ob_low"],
        "current_bearish_ob_high": ict_smc_features["current_bearish_ob_high"],
        "last_touched_ob_type": ict_smc_features["last_touched_ob_type"],
        "has_mitigation": ict_smc_features["has_mitigation"],
        "has_flip_zone": ict_smc_features["has_flip_zone"],
        "mitigation_low": ict_smc_features["mitigation_low"],
        "mitigation_high": ict_smc_features["mitigation_high"],
        "flip_low": ict_smc_features["flip_low"],
        "flip_high": ict_smc_features["flip_high"],
        "has_fvg": ict_smc_features["has_fvg"],
        "fvg_upper": ict_smc_features["fvg_upper"],
        "fvg_lower": ict_smc_features["fvg_lower"],
        "has_ifvg": ict_smc_features["has_ifvg"],
        "ifvg_upper": ict_smc_features["ifvg_upper"],
        "ifvg_lower": ict_smc_features["ifvg_lower"],
        "premium_discount_state": ict_smc_features["premium_discount_state"],
        "equilibrium_level": ict_smc_features["equilibrium_level"],
        "in_london_killzone": ict_smc_features["in_london_killzone"],
        "in_ny_killzone": ict_smc_features["in_ny_killzone"],
        "p_sweep_reversal": pattern_probs.p_sweep_reversal,
        "p_sweep_continuation": pattern_probs.p_sweep_continuation,
        "p_ob_hold": pattern_probs.p_ob_hold,
        "p_ob_fail": pattern_probs.p_ob_fail,
        "p_fvg_fill": pattern_probs.p_fvg_fill,
        "expected_move_after_sweep": pattern_probs.expected_move_after_sweep,
        "expected_move_after_ob_touch": pattern_probs.expected_move_after_ob_touch,
        "expected_move_after_fvg_fill": pattern_probs.expected_move_after_fvg_fill,
        "bar_delta": orderflow_metrics["bar_delta"],
        "cumulative_delta": orderflow_metrics["cumulative_delta"],
        "footprint_imbalance": orderflow_metrics["footprint_imbalance"],
        "has_absorption": orderflow_metrics["has_absorption"],
        "absorption_side": orderflow_metrics["absorption_side"],
        "has_exhaustion": orderflow_metrics["has_exhaustion"],
        "exhaustion_side": orderflow_metrics["exhaustion_side"],
        "ema_9": trend_indicator["ema_9"],
        "ema_20": trend_indicator["ema_20"],
        "ema_50": trend_indicator["ema_50"],
        "ema_200": trend_indicator["ema_200"],
        "sma_20": trend_indicator["sma_20"],
        "sma_50": trend_indicator["sma_50"],
        "sma_200": trend_indicator["sma_200"],
        "ma_stack_state": trend_indicator["ma_stack_state"],
        "distance_from_ema_20": trend_indicator["distance_from_ema_20"],
        "distance_from_ema_50": trend_indicator["distance_from_ema_50"],
        "momentum_5": momentum_features["momentum_5"],
        "momentum_10": momentum_features["momentum_10"],
        "momentum_20": momentum_features["momentum_20"],
        "roc_5": momentum_features["roc_5"],
        "roc_10": momentum_features["roc_10"],
        "roc_20": momentum_features["roc_20"],
        "raw": {"governance": governance_metadata},
        "timestamp": timestamp,
        "health": {"is_stale": False, "errors": []},
    }
    return state
