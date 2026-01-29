"""
Validation and clamping for Ollama Gateway outputs.
Keeps all values engine-safe and bounded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ollama_gateway import config

REQUIRED_FIELDS = {
    "source",
    "origin_id",
    "timestamp",
    "impact_level",
    "directional_bias",
    "confidence",
    "sentiment_score",
    "sentiment_volatility",
}

IMPACT_ORDER = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 1.0}
BIAS_MAP = {"UP": 1.0, "DOWN": -1.0, "NEUTRAL": 0.0}


def _clamp(val: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(val)))
    except Exception:
        return low


def _clean_list(items: Any, limit: int = 8) -> List[str]:
    if not isinstance(items, (list, tuple)):
        return []
    out: List[str] = []
    for it in items[:limit]:
        try:
            out.append(str(it)[:80])
        except Exception:
            continue
    return out


def validate_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    if not REQUIRED_FIELDS.issubset(record.keys()):
        return None

    ts_raw = str(record.get("timestamp", ""))
    try:
        datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        ts_val = ts_raw
    except Exception:
        ts_val = datetime.now(tz=timezone.utc).isoformat()

    impact = str(record.get("impact_level", "MEDIUM")).upper()
    if impact not in config.IMPACT_LEVELS:
        impact = "MEDIUM"

    bias = str(record.get("directional_bias", "NEUTRAL")).upper()
    if bias not in config.DIRECTIONAL_BIAS:
        bias = "NEUTRAL"

    clean_record = {
        "source": str(record.get("source", "forex")),
        "origin_id": str(record.get("origin_id", "unknown"))[:128],
        "timestamp": ts_val,
        "asset_scope": _clean_list(record.get("asset_scope", []), limit=6),
        "event_type": str(record.get("event_type", "OTHER"))[:32],
        "impact_level": impact,
        "directional_bias": bias,
        "confidence": _clamp(record.get("confidence", 0.0), 0.0, 1.0),
        "sentiment_score": _clamp(record.get("sentiment_score", 0.0), -1.0, 1.0),
        "sentiment_volatility": _clamp(
            record.get("sentiment_volatility", 0.0), 0.0, 1.0
        ),
        "summary": str(record.get("summary", ""))[:320],
        "keywords": _clean_list(record.get("keywords", [])),
        "numeric_extractions": {},
    }

    for k, v in (record.get("numeric_extractions") or {}).items():
        try:
            clean_record["numeric_extractions"][str(k)[:64]] = float(v)
        except Exception:
            continue

    return clean_record


def build_snapshot(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    validated = [r for r in (validate_record(rec) for rec in records) if r]
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "record_count": len(validated),
        "records": validated,
    }


def aggregate_features(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {
            "news_sentiment_score": 0.0,
            "news_sentiment_volatility": 0.0,
            "news_macro_impact": 0.0,
            "news_directional_bias": 0.0,
            "news_confidence": 0.0,
            # requested field names
            "sentiment_score": 0.0,
            "sentiment_volatility": 0.0,
            "impact": "LOW",
            "directional_bias": "NEUTRAL",
            "confidence": 0.0,
        }

    scores = [float(r.get("sentiment_score", 0.0)) for r in records]
    confs = [float(r.get("confidence", 0.0)) for r in records]
    impacts = [
        (
            str(r.get("impact_level", "MEDIUM")).upper(),
            IMPACT_ORDER.get(str(r.get("impact_level", "MEDIUM")).upper(), 0.6),
        )
        for r in records
    ]
    bias_values = [
        (
            str(r.get("directional_bias", "NEUTRAL")).upper(),
            BIAS_MAP.get(str(r.get("directional_bias", "NEUTRAL")).upper(), 0.0),
        )
        for r in records
    ]

    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    # pick max impact level by weight
    impact_level = max(impacts, key=lambda x: x[1])[0] if impacts else "LOW"
    mean_impact = max([i[1] for i in impacts]) if impacts else 0.0
    mean_bias_val = (
        sum([b[1] for b in bias_values]) / len(bias_values) if bias_values else 0.0
    )
    bias_label = (
        max(bias_values, key=lambda x: abs(x[1]))[0] if bias_values else "NEUTRAL"
    )

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
        "news_directional_bias": _clamp(mean_bias_val, -1.0, 1.0),
        "news_confidence": _clamp(mean_conf, 0.0, 1.0),
        "sentiment_score": _clamp(mean_score, -1.0, 1.0),
        "sentiment_volatility": _clamp(vol, 0.0, 1.0),
        "impact": impact_level,
        "directional_bias": bias_label,
        "confidence": _clamp(mean_conf, 0.0, 1.0),
    }
