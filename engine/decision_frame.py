from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional

from .structure_brain import LiquidityFrame, MarketProfileFrame, SessionProfileFrame


@dataclass
class DecisionFrame:
    timestamp_utc: Optional[str] = None
    symbol: Optional[str] = None
    session_context: Optional[Dict[str, Any]] = None
    condition_vector: Optional[Dict[str, Any]] = None

    vol_regime: Optional[str] = None
    trend_regime: Optional[str] = None

    market_profile_state: Optional[str] = None
    market_profile_confidence: Optional[float] = None
    market_profile_evidence: Optional[Dict[str, Any]] = None

    session_profile: Optional[str] = None
    session_profile_confidence: Optional[float] = None
    session_profile_evidence: Optional[Dict[str, Any]] = None

    liquidity_frame: Optional[Dict[str, Any]] = None

    entry_signals_present: Dict[str, Any] = None
    eligible_entry_models: List[str] = None
    chosen_entry_model_id: Optional[str] = None

    risk_per_trade: Optional[float] = None
    position_size: Optional[float] = None

    entry_brain_labels: Optional[Dict[str, Any]] = None
    entry_brain_scores: Optional[Dict[str, Any]] = None
    entry_consistency_report: Optional[Dict[str, Any]] = None

    @staticmethod
    def _serialize(value: Any) -> Any:
        if value is None:
            return None
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return dict(value)
        return value

    @classmethod
    def from_frames(
        cls,
        *,
        timestamp_utc: Optional[str] = None,
        symbol: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
        condition_vector: Optional[Dict[str, Any]] = None,
        market_profile: Optional[MarketProfileFrame] = None,
        session_profile: Optional[SessionProfileFrame] = None,
        liquidity: Optional[LiquidityFrame] = None,
    ) -> "DecisionFrame":
        frame = cls(
            timestamp_utc=timestamp_utc,
            symbol=symbol,
            session_context=session_context,
            condition_vector=cls._serialize(condition_vector),
        )
        if market_profile is not None:
            frame.market_profile_state = getattr(market_profile, "state", None)
            frame.market_profile_confidence = getattr(
                market_profile, "confidence", None
            )
            frame.market_profile_evidence = cls._serialize(
                getattr(market_profile, "evidence", None)
            )
        if session_profile is not None:
            frame.session_profile = getattr(session_profile, "profile", None)
            frame.session_profile_confidence = getattr(
                session_profile, "confidence", None
            )
            frame.session_profile_evidence = cls._serialize(
                getattr(session_profile, "evidence", None)
            )
        if liquidity is not None:
            frame.liquidity_frame = cls._serialize(liquidity)
        return frame

    def to_dict(self) -> Dict[str, Any]:
        # Deterministic ordering; only JSON-serializable primitives/dicts
        ordered_keys = [
            "timestamp_utc",
            "symbol",
            "session_context",
            "condition_vector",
            "vol_regime",
            "trend_regime",
            "market_profile_state",
            "market_profile_confidence",
            "market_profile_evidence",
            "session_profile",
            "session_profile_confidence",
            "session_profile_evidence",
            "liquidity_frame",
            "entry_signals_present",
            "eligible_entry_models",
            "chosen_entry_model_id",
            "risk_per_trade",
            "position_size",
            "entry_brain_labels",
            "entry_brain_scores",
            "entry_consistency_report",
        ]

        output: Dict[str, Any] = {}
        for key in ordered_keys:
            value = getattr(self, key)
            output[key] = self._serialize(value)
        return output
