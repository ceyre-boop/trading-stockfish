"""
Market State Builder v4.0‑C
Injects liquidity block into MarketState.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from bayesian_probability_engine import compute_bayesian_probabilities
from candle_pattern_features import compute_candle_pattern_features
from engine.amd_features import AMDFeatures
from engine.liquidity_features import LiquidityFeatures
from engine.macro_bayes_model import adjust_bayesian_priors
from engine.macro_risk_model import compute_macro_risk_adjustment
from engine.macro_search_model import compute_search_modulation
from engine.order_book_model import OrderBookModel
from engine.order_flow_features import OrderFlowFeatures
from engine.trend_structure import compute_trend_structure
from engine.volatility_utils import compute_atr
from ict_smc_features import compute_ict_smc_features
from levels_features import compute_level_features
from liquidity_depth_features import compute_depth_features
from liquidity_primitives import compute_liquidity_primitives
from momentum_features import compute_momentum_features
from momentum_indicators_v2 import compute_momentum_indicators
from mtf_structure_features import compute_mtf_structure_features
from news_macro_features import compute_news_macro_features
from orderbook_features import OrderBookFeaturesConfig, compute_orderbook_features
from orderflow_features import compute_orderflow_features
from pattern_stats import compute_pattern_probabilities
from session_regime import SessionRegime, compute_session_regime
from structure_features import compute_structure_features
from trend_indicator_features import compute_trend_indicator_features
from volume_profile_features import VolumeProfileConfig, compute_volume_profile_features


class LiquidityState:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.__dict__


# NOTE: All news ingestion must flow through ollama_gateway/outbox/news_snapshot.json.
# Do NOT bypass this file with raw RSS/Forex/Twitter text; the engine only consumes
# validated, clamped JSON produced by the gateway.
GATEWAY_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ollama_gateway"
    / "outbox"
    / "news_snapshot.json"
)


def _load_gateway_snapshot(
    snapshot: Optional[Dict[str, Any]], snapshot_path: Optional[str]
) -> Dict[str, Any]:
    """All macro/news understanding must flow from the validated gateway snapshot."""
    if isinstance(snapshot, dict):
        return snapshot
    path = Path(snapshot_path) if snapshot_path else GATEWAY_SNAPSHOT_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _aggregate_news_features(records: List[Dict[str, Any]]) -> Dict[str, float]:
    def _clamp(val: float, low: float, high: float) -> float:
        try:
            return max(low, min(high, float(val)))
        except Exception:
            return low

    bias_map = {"UP": 1.0, "DOWN": -1.0, "NEUTRAL": 0.0}
    impact_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 1.0}

    scores = [float(r.get("sentiment_score", 0.0)) for r in records]
    confs = [float(r.get("confidence", 0.0)) for r in records]
    impacts = [
        impact_map.get(str(r.get("impact_level", "MEDIUM")).upper(), 0.6)
        for r in records
    ]
    biases = [
        bias_map.get(str(r.get("directional_bias", "NEUTRAL")).upper(), 0.0)
        for r in records
    ]

    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    mean_impact = max(impacts) if impacts else 0.0
    mean_bias = sum(biases) / len(biases) if biases else 0.0

    if len(scores) > 1:
        mean = mean_score
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        vol = var**0.5
    else:
        vol = 0.0

    return {
        "news_sentiment_score": _clamp(mean_score, -1.0, 1.0),
        "news_sentiment_volatility": _clamp(vol, 0.0, 1.0),
        "news_macro_impact": _clamp(mean_impact, 0.0, 1.0),
        "news_directional_bias": _clamp(mean_bias, -1.0, 1.0),
        "news_confidence": _clamp(mean_conf, 0.0, 1.0),
    }


def _filter_records(records: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    return [r for r in records if str(r.get("source", "")).lower() == source]


def _future_event_metrics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    future = snapshot.get("future_events", []) if isinstance(snapshot, dict) else []
    if not isinstance(future, list):
        future = []
    macro_pressure = max(
        [float(e.get("macro_pressure_score", 0.0)) for e in future], default=0.0
    )
    next_event = future[0] if future else {}
    return {
        "future_events_count": len(future),
        "next_event_type": next_event.get("event_type", "NONE"),
        "next_event_time_delta": float(
            next_event.get("time_delta_minutes", 0.0) or 0.0
        ),
        "next_event_impact": next_event.get("impact_level", "NONE"),
        "macro_pressure_score": macro_pressure,
    }


def build_market_state(
    symbol: str,
    order_book_events: Optional[List[Dict]] = None,
    use_microstructure_realism: bool = True,
    timestamp: Optional[float] = None,
    event_calendar: Optional[List[Dict]] = None,
    macro_inputs: Optional[Dict[str, Any]] = None,
    sentiment_snapshot: Optional[Dict[str, Any]] = None,
    llm_news_snapshot: Optional[Dict[str, Any]] = None,
    twitter_news_snapshot: Optional[Dict[str, Any]] = None,
    news_ollama_snapshot: Optional[Dict[str, Any]] = None,
    news_snapshot_path: Optional[str] = None,
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
    spreads_series: List[float] = []

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
            spreads_series.append(best_ask - best_bid)
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

    current_time = (
        timestamp
        if timestamp is not None
        else (timestamps_series[-1] if timestamps_series else 0.0)
    )
    news_macro_state = compute_news_macro_features(
        current_time=current_time,
        calendar_data=event_calendar,
        macro_inputs=macro_inputs,
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
    momentum_indicators_v2 = compute_momentum_indicators(
        mid_prices,
        volatility=(
            volatility_state.get("realized_vol", 0.0)
            if isinstance(volatility_state, dict)
            else 0.0
        ),
        trend_strength=trend_struct.get("trend_strength", 0.0),
    )
    mtf_features = compute_mtf_structure_features(
        mid_prices,
        timestamps_series,
        timeframes=("1H", "4H", "D"),
        ltf_trend_direction=trend_struct["trend_direction"],
        ltf_trend_strength=trend_indicator["trend_strength"],
        ltf_volatility=(
            volatility_state.get("realized_vol", 0.0)
            if isinstance(volatility_state, dict)
            else 0.0
        ),
    )
    candle_pattern_features = compute_candle_pattern_features(
        mid_prices,
        volumes,
        context={
            "bsl_zone_price": liquidity_primitives.get("bsl_zone_price"),
            "ssl_zone_price": liquidity_primitives.get("ssl_zone_price"),
            "nearest_bsl_pool_above": liquidity_primitives.get(
                "nearest_bsl_pool_above"
            ),
            "nearest_ssl_pool_below": liquidity_primitives.get(
                "nearest_ssl_pool_below"
            ),
            "swing_high": trend_struct.get("swing_high"),
            "swing_low": trend_struct.get("swing_low"),
            "current_bullish_ob_high": ict_smc_features.get("current_bullish_ob_high"),
            "current_bullish_ob_low": ict_smc_features.get("current_bullish_ob_low"),
            "current_bearish_ob_high": ict_smc_features.get("current_bearish_ob_high"),
            "current_bearish_ob_low": ict_smc_features.get("current_bearish_ob_low"),
        },
    )

    volume_profile_config = VolumeProfileConfig(
        price_bin_size=max(abs(mid_prices[-1]) * 0.001, 0.5) if mid_prices else 1.0,
        range_mode="ROLLING_WINDOW",
        rolling_window_size=200,
        value_area_target=0.7,
        hvn_threshold_ratio=0.7,
        lvn_threshold_ratio=0.3,
        proximity_band=1.5,
    )
    volume_profile_features = compute_volume_profile_features(
        mid_prices,
        price_volumes,
        config=volume_profile_config,
    )

    orderbook_config = OrderBookFeaturesConfig(
        max_levels_per_side=10,
        aggregation_mode="RAW",
        tick_size=0.01,
    )
    recent_aggr_buy = sum(aggressive_buy_series[-5:]) if aggressive_buy_series else 0.0
    recent_aggr_sell = (
        sum(aggressive_sell_series[-5:]) if aggressive_sell_series else 0.0
    )
    orderbook_features = compute_orderbook_features(
        ob_snapshot,
        config=orderbook_config,
        recent_spreads=spreads_series,
        aggressive_buy=recent_aggr_buy,
        aggressive_sell=recent_aggr_sell,
        prev_snapshot=None,
    )

    # Prepare a thin state view for Bayesian updating
    bayes_inputs: Dict[str, Any] = {
        "trend_strength": trend_struct.get("trend_strength", 0.0),
        "momentum_regime": momentum_indicators_v2.get("momentum_regime", "CHOP"),
        "rsi_bullish_divergence": momentum_indicators_v2.get(
            "rsi_bullish_divergence", False
        ),
        "rsi_bearish_divergence": momentum_indicators_v2.get(
            "rsi_bearish_divergence", False
        ),
        "macd_bullish_divergence": momentum_indicators_v2.get(
            "macd_bullish_divergence", False
        ),
        "macd_bearish_divergence": momentum_indicators_v2.get(
            "macd_bearish_divergence", False
        ),
        "last_sweep_direction": liquidity_primitives.get(
            "last_sweep_direction", "NONE"
        ),
        "has_absorption": orderflow_metrics.get("has_absorption", False),
        "has_exhaustion": orderflow_metrics.get("has_exhaustion", False),
        "footprint_imbalance": orderflow_metrics.get("footprint_imbalance", 0.0),
        "last_touched_ob_type": ict_smc_features.get("last_touched_ob_type", "NONE"),
        "has_mitigation": ict_smc_features.get("has_mitigation", False),
        "has_flip_zone": ict_smc_features.get("has_flip_zone", False),
        "has_fvg": ict_smc_features.get("has_fvg", False),
        "expected_volatility_state": news_macro_state.get(
            "expected_volatility_state", "LOW"
        ),
        "volatility_regime": (
            volatility_state.get("vol_regime", "NORMAL")
            if isinstance(volatility_state, dict)
            else "NORMAL"
        ),
        "p_sweep_reversal": pattern_probs.p_sweep_reversal,
        "p_sweep_continuation": pattern_probs.p_sweep_continuation,
        "p_ob_hold": pattern_probs.p_ob_hold,
        "p_ob_fail": pattern_probs.p_ob_fail,
        "p_fvg_fill": pattern_probs.p_fvg_fill,
    }

    bayesian_state = compute_bayesian_probabilities(bayes_inputs)

    # Governance metadata (dummy for now, to be updated by engine)
    governance_metadata = {
        "recent_vetoes": [],
        "drawdown": 0.0,
        "trade_frequency": 0,
        "regime_transition": regime_state.get("regime_transition", False),
    }

    sentiment_score = 0.0
    sentiment_volatility = 0.0
    if sentiment_snapshot:
        try:
            sentiment_score = float(sentiment_snapshot.get("sentiment_score", 0.0))
            sentiment_volatility = float(
                sentiment_snapshot.get("sentiment_volatility", 0.0)
            )
        except Exception:
            sentiment_score = 0.0
            sentiment_volatility = 0.0
    # Enforce gateway-only ingestion. Any direct llm_news_snapshot or
    # twitter_news_snapshot inputs are ignored unless they already came from
    # the gateway. All validated records should live in news_snapshot.
    news_snapshot = _load_gateway_snapshot(news_ollama_snapshot, news_snapshot_path)
    parsed_records = (
        news_snapshot.get("parsed_events", [])
        if isinstance(news_snapshot, dict)
        else []
    )
    aggregated_scores = (
        news_snapshot.get("aggregated_scores", {})
        if isinstance(news_snapshot, dict)
        else {}
    )
    ollama_unreachable = bool(
        news_snapshot.get("ollama_unreachable", False)
        if isinstance(news_snapshot, dict)
        else False
    )

    def _score(key: str, default: float = 0.0) -> float:
        try:
            return float(aggregated_scores.get(key, default))
        except Exception:
            return default

    news_features = {
        "news_sentiment_score": _score(
            "news_sentiment_score", _score("sentiment_score", 0.0)
        ),
        "news_sentiment_volatility": _score(
            "news_sentiment_volatility", _score("sentiment_volatility", 0.0)
        ),
        "news_macro_impact": _score("news_macro_impact", 0.0),
        "news_directional_bias": _score("news_directional_bias", 0.0),
        "news_confidence": _score("news_confidence", _score("confidence", 0.0)),
    }

    if not aggregated_scores:
        news_features = _aggregate_news_features(parsed_records)

    twitter_records = _filter_records(parsed_records, "twitter")
    twitter_features = _aggregate_news_features(twitter_records)
    future_metrics = _future_event_metrics(news_snapshot)
    risk_window_label = (
        "RISK_WINDOW"
        if abs(future_metrics.get("next_event_time_delta", 9999)) <= 30
        else "NONE"
    )

    macro_risk = compute_macro_risk_adjustment(
        future_metrics["macro_pressure_score"],
        future_metrics["next_event_time_delta"],
        future_metrics["next_event_impact"],
    )

    base_priors = {
        "trend_continuation": 0.4,
        "mean_reversion": 0.3,
        "breakout": 0.3,
    }
    macro_adjusted_priors = adjust_bayesian_priors(
        base_priors,
        future_metrics["macro_pressure_score"],
        future_metrics["next_event_impact"],
    )

    search_modulation = compute_search_modulation(
        future_metrics["macro_pressure_score"], future_metrics["next_event_time_delta"]
    )

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
        "sentiment_score": sentiment_score,
        "sentiment_volatility": sentiment_volatility,
        "news_sentiment_score": news_features["news_sentiment_score"],
        "news_sentiment_volatility": news_features["news_sentiment_volatility"],
        "news_macro_impact": news_features["news_macro_impact"],
        "news_impact": news_features["news_macro_impact"],
        "news_directional_bias": news_features["news_directional_bias"],
        "news_confidence": news_features["news_confidence"],
        "news_snapshot": news_snapshot,
        "ollama_unreachable": ollama_unreachable,
        "future_events_count": future_metrics["future_events_count"],
        "parsed_events_count": len(parsed_records),
        "next_event_type": future_metrics["next_event_type"],
        "next_event_time_delta": future_metrics["next_event_time_delta"],
        "next_event_impact": future_metrics["next_event_impact"],
        "macro_pressure_score": future_metrics["macro_pressure_score"],
        "macro_entry_allowed": macro_risk["entry_allowed"],
        "macro_position_size_multiplier": macro_risk["position_size_multiplier"],
        "macro_max_leverage_multiplier": macro_risk["max_leverage_multiplier"],
        "macro_adjusted_priors": macro_adjusted_priors,
        "macro_search_depth_multiplier": search_modulation["search_depth_multiplier"],
        "macro_aggressiveness_bias": search_modulation["aggressiveness_bias"],
        "event_risk_window": risk_window_label,
        "twitter_sentiment_score": twitter_features["news_sentiment_score"],
        "twitter_sentiment_volatility": twitter_features["news_sentiment_volatility"],
        "twitter_news_snapshot": {
            "count": len(twitter_records),
            "records": twitter_records,
        },
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
        "rsi_value": momentum_indicators_v2["rsi_value"],
        "rsi_state": momentum_indicators_v2["rsi_state"],
        "macd_value": momentum_indicators_v2["macd_value"],
        "macd_signal": momentum_indicators_v2["macd_signal"],
        "macd_histogram": momentum_indicators_v2["macd_histogram"],
        "macd_state": momentum_indicators_v2["macd_state"],
        "stoch_k": momentum_indicators_v2["stoch_k"],
        "stoch_d": momentum_indicators_v2["stoch_d"],
        "stoch_state": momentum_indicators_v2["stoch_state"],
        "momentum_regime": momentum_indicators_v2["momentum_regime"],
        "momentum_confidence": momentum_indicators_v2["momentum_confidence"],
        "rsi_bullish_divergence": momentum_indicators_v2["rsi_bullish_divergence"],
        "rsi_bearish_divergence": momentum_indicators_v2["rsi_bearish_divergence"],
        "macd_bullish_divergence": momentum_indicators_v2["macd_bullish_divergence"],
        "macd_bearish_divergence": momentum_indicators_v2["macd_bearish_divergence"],
        "next_event_type": news_macro_state["next_event_type"],
        "next_event_time_delta": news_macro_state["next_event_time_delta"],
        "event_risk_window": news_macro_state["event_risk_window"],
        "expected_volatility_state": news_macro_state["expected_volatility_state"],
        "expected_volatility_score": news_macro_state["expected_volatility_score"],
        "liquidity_withdrawal_flag": news_macro_state["liquidity_withdrawal_flag"],
        "macro_regime": news_macro_state["macro_regime"],
        "macro_regime_score": news_macro_state["macro_regime_score"],
        "bayes_trend_continuation": bayesian_state["bayes_trend_continuation"],
        "bayes_trend_continuation_confidence": bayesian_state[
            "bayes_trend_continuation_confidence"
        ],
        "bayes_trend_reversal": bayesian_state["bayes_trend_reversal"],
        "bayes_trend_reversal_confidence": bayesian_state[
            "bayes_trend_reversal_confidence"
        ],
        "bayes_sweep_reversal": bayesian_state["bayes_sweep_reversal"],
        "bayes_sweep_reversal_confidence": bayesian_state[
            "bayes_sweep_reversal_confidence"
        ],
        "bayes_sweep_continuation": bayesian_state["bayes_sweep_continuation"],
        "bayes_sweep_continuation_confidence": bayesian_state[
            "bayes_sweep_continuation_confidence"
        ],
        "bayes_ob_respect": bayesian_state["bayes_ob_respect"],
        "bayes_ob_respect_confidence": bayesian_state["bayes_ob_respect_confidence"],
        "bayes_ob_violation": bayesian_state["bayes_ob_violation"],
        "bayes_ob_violation_confidence": bayesian_state[
            "bayes_ob_violation_confidence"
        ],
        "bayes_fvg_fill": bayesian_state["bayes_fvg_fill"],
        "bayes_fvg_fill_confidence": bayesian_state["bayes_fvg_fill_confidence"],
        "bayes_fvg_reject": bayesian_state["bayes_fvg_reject"],
        "bayes_fvg_reject_confidence": bayesian_state["bayes_fvg_reject_confidence"],
        "bayesian_update_strength": bayesian_state["bayesian_update_strength"],
        "body_size": candle_pattern_features["body_size"],
        "upper_wick_size": candle_pattern_features["upper_wick_size"],
        "lower_wick_size": candle_pattern_features["lower_wick_size"],
        "total_range": candle_pattern_features["total_range"],
        "wick_to_body_upper": candle_pattern_features["wick_to_body_upper"],
        "wick_to_body_lower": candle_pattern_features["wick_to_body_lower"],
        "wick_to_body_total": candle_pattern_features["wick_to_body_total"],
        "bullish_engulfing": candle_pattern_features["bullish_engulfing"],
        "bearish_engulfing": candle_pattern_features["bearish_engulfing"],
        "inside_bar": candle_pattern_features["inside_bar"],
        "outside_bar": candle_pattern_features["outside_bar"],
        "pin_bar_upper": candle_pattern_features["pin_bar_upper"],
        "pin_bar_lower": candle_pattern_features["pin_bar_lower"],
        "momentum_bar": candle_pattern_features["momentum_bar"],
        "exhaustion_bar": candle_pattern_features["exhaustion_bar"],
        "high_volume_candle": candle_pattern_features["high_volume_candle"],
        "low_volume_candle": candle_pattern_features["low_volume_candle"],
        "pattern_at_liquidity": candle_pattern_features["pattern_at_liquidity"],
        "pattern_at_structure": candle_pattern_features["pattern_at_structure"],
        "pattern_context_importance": candle_pattern_features[
            "pattern_context_importance"
        ],
        "l2_bids": orderbook_features["l2_bids"],
        "l2_asks": orderbook_features["l2_asks"],
        "top_level_imbalance": orderbook_features["top_level_imbalance"],
        "multi_level_imbalance": orderbook_features["multi_level_imbalance"],
        "spread_ticks": orderbook_features["spread_ticks"],
        "microstructure_shift": orderbook_features["microstructure_shift"],
        "spread_widening": orderbook_features["spread_widening"],
        "spread_tightening": orderbook_features["spread_tightening"],
        "hidden_bid_liquidity": orderbook_features["hidden_bid_liquidity"],
        "hidden_ask_liquidity": orderbook_features["hidden_ask_liquidity"],
        "queue_position_estimate": orderbook_features["queue_position_estimate"],
        "poc_price": volume_profile_features["poc_price"],
        "hvn_levels": volume_profile_features["hvn_levels"],
        "lvn_levels": volume_profile_features["lvn_levels"],
        "value_area_low": volume_profile_features["value_area_low"],
        "value_area_high": volume_profile_features["value_area_high"],
        "value_area_coverage": volume_profile_features["value_area_coverage"],
        "price_vs_value_area_state": volume_profile_features[
            "price_vs_value_area_state"
        ],
        "near_hvn": volume_profile_features["near_hvn"],
        "near_lvn": volume_profile_features["near_lvn"],
        "raw": {"governance": governance_metadata},
        "timestamp": timestamp,
        "health": {"is_stale": False, "errors": []},
    }
    state.update(mtf_features)
    return state
