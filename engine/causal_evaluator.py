"""
CAUSAL EVALUATOR: Stockfish-Style Trading Engine Evaluation
============================================================

A deterministic, rule-based evaluation system that consumes all 8 market state
variables and produces a unified EvalScore [-1, +1] with confidence and causal
reasoning, similar to how chess engines evaluate positions.

Philosophy:
- Stockfish evaluates chess positions by scoring multiple factors (material,
  piece mobility, king safety, pawn structure, etc.) and combining them.
- Our CausalEvaluator does the same for markets: score macro conditions,
  liquidity, volatility, dealer positioning, earnings, time regime, price
  location, and news/macro factors, then combine them.

Key Principles:
1. DETERMINISTIC: No ML, no stochasticity, pure rule-based logic
2. CAUSAL: Each score reflects actual market mechanics
3. TIME-CAUSAL: All data is real and past-only
4. EXPLAINABLE: Full reasoning for every evaluation
5. CONFIGURABLE: Weights are tunable, not hard-coded

Author: Trading Stockfish Engine
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engine.canonical_stack_validator import validate_official_mode_startup
from engine.canonical_validator import assert_causal_required, enforce_official_env
from state.regime_engine import RegimeSignal

# Canonical state + regime signal
from state.schema import MarketState as SchemaMarketState

# Import regime classifier for regime-conditioned evaluation
try:
    from engine.regime_classifier import RegimeClassifier, RegimeState

    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False

# Import scenario simulator for regime-aware scenario generation (v2.2)
try:
    from engine.scenario_simulator import ScenarioResult, ScenarioSimulator

    SCENARIO_AVAILABLE = True
except ImportError:
    SCENARIO_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Regime-conditioned factor weights (deterministic, config-style constants)
FACTOR_WEIGHTS = {
    "VOL": {
        "LOW": 0.2,
        "NORMAL": 0.3,
        "HIGH": 0.4,
        "EXTREME": 0.5,
    },
    "LIQ": {
        "DEEP": 0.35,
        "NORMAL": 0.30,
        "THIN": 0.20,
        "FRAGILE": 0.10,
    },
    "MACRO": {
        "RISK_ON": 0.30,
        "RISK_OFF": 0.25,
        "EVENT": 0.20,
    },
}

# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================


class MacroTrend(Enum):
    """Macro trend classifications."""

    STRONGLY_DOVISH = -1.0
    DOVISH = -0.5
    NEUTRAL = 0.0
    HAWKISH = 0.5
    STRONGLY_HAWKISH = 1.0


class LiquidityRegime(Enum):
    """Liquidity regime classifications."""

    ABSORBING = 1.0  # Price absorption, continuation bias
    NORMAL = 0.0  # Normal liquidity
    EXHAUSTING = -1.0  # Liquidity exhaustion, reversal bias


class VolatilityRegime(Enum):
    """Volatility regime classifications."""

    EXPANDING = 1.0  # Vol expansion, trend continuation
    NORMAL = 0.0  # Normal volatility
    COMPRESSING = -1.0  # Vol compression, breakout risk


class TimeRegimeType(Enum):
    """Time regime classifications."""

    ASIAN_EARLY = -0.3
    ASIAN_LATE = -0.1
    LONDON_OPEN = 0.3
    LONDON_CLOSE = 0.2
    NY_OPEN = 0.5  # Highest volatility, directional risk
    POWER_HOUR = 0.4  # Trend continuation hour
    NY_CLOSE = 0.2
    OVERNIGHT = 0.0


class RiskSentiment(Enum):
    """Overall risk sentiment from news/macro."""

    STRONG_RISK_OFF = -1.0
    MILD_RISK_OFF = -0.5
    NEUTRAL = 0.0
    MILD_RISK_ON = 0.5
    STRONG_RISK_ON = 1.0


# =============================================================================
# MARKET STATE DATACLASSES (expected input structure)
# =============================================================================


@dataclass
class MacroState:
    """Macro expectation state."""

    sentiment_score: float  # [-1, 1] dovish to hawkish
    surprise_score: float  # [-1, 1] miss to beat
    rate_expectation: float  # [-1, 1] cut to hike
    inflation_expectation: float  # [-1, 1] deflation to inflation
    gdp_expectation: float  # [-1, 1] contraction to expansion

    def __post_init__(self):
        assert -1.0 <= self.sentiment_score <= 1.0
        assert -1.0 <= self.surprise_score <= 1.0
        assert -1.0 <= self.rate_expectation <= 1.0


@dataclass
class LiquidityState:
    """Liquidity market state."""

    bid_ask_spread: float  # Basis points
    order_book_depth: float  # Aggregate depth (normalized 0-1)
    regime: LiquidityRegime  # Absorbing/Normal/Exhausting
    volume_trend: float  # [-1, 1] declining to expanding

    def __post_init__(self):
        assert self.bid_ask_spread >= 0


@dataclass
class VolatilityState:
    """Volatility market state."""

    current_vol: float  # Current volatility (annualized)
    vol_percentile: float  # [0, 1] percentile of historical
    regime: VolatilityRegime  # Expanding/Normal/Compressing
    vol_trend: float  # [-1, 1] contracting to expanding
    skew: float  # [-1, 1] downside to upside skew

    def __post_init__(self):
        assert 0.0 <= self.vol_percentile <= 1.0


@dataclass
class DealerState:
    """Dealer positioning state."""

    net_gamma_exposure: float  # [-1, 1] negative to positive gamma
    net_spot_exposure: float  # [-1, 1] short to long spot
    vega_exposure: float  # [-1, 1] short to long vega
    dealer_sentiment: float  # [-1, 1] bearish to bullish

    def __post_init__(self):
        assert -1.0 <= self.net_gamma_exposure <= 1.0


@dataclass
class EarningsState:
    """Earnings exposure state."""

    multi_mega_cap_exposure: float  # [0, 1] NQ component exposure
    small_cap_exposure: float  # [0, 1] Russell 2000 component
    earnings_season_flag: bool  # True if in earnings season
    earnings_surprise_momentum: float  # [-1, 1] miss to beat trend

    def __post_init__(self):
        assert 0.0 <= self.multi_mega_cap_exposure <= 1.0
        assert 0.0 <= self.small_cap_exposure <= 1.0


@dataclass
class TimeRegimeState:
    """Time regime state."""

    regime_type: TimeRegimeType  # Session type
    minutes_into_session: int  # Minutes since session start
    hours_until_session_end: float  # Hours until close
    day_of_week: int  # 0=Monday, 4=Friday

    def __post_init__(self):
        assert 0 <= self.day_of_week <= 4
        assert self.minutes_into_session >= 0


@dataclass
class PriceLocationState:
    """Price location within session/range."""

    distance_from_high: float  # [0, 1] at low to at high
    distance_from_low: float  # [0, 1] at high to at low
    range_ratio: float  # Current range / average range
    session_extremity: float  # [-1, 1] at low to at high

    def __post_init__(self):
        assert 0.0 <= self.distance_from_high <= 1.0
        assert 0.0 <= self.distance_from_low <= 1.0


@dataclass
class MacroNewsState:
    """Macro news features (from NewsMaproEngine)."""

    risk_sentiment_score: float  # [-1, 1] risk-off to risk-on
    hawkishness_score: float  # [-1, 1] dovish to hawkish
    surprise_score: float  # [-1, 1] miss to beat
    event_importance: int  # [0, 3] impact level
    hours_since_last_event: float  # Recency weight
    macro_event_count: int  # Recent event frequency
    news_article_count: int  # Recent article count
    macro_news_state: str  # STRONG_RISK_ON, NEUTRAL, etc.

    def __post_init__(self):
        assert -1.0 <= self.risk_sentiment_score <= 1.0
        assert -1.0 <= self.hawkishness_score <= 1.0
        assert -1.0 <= self.surprise_score <= 1.0


@dataclass
class MarketState:
    """Complete market state (all 8 components)."""

    timestamp: datetime
    symbol: str

    # 8 core market state variables
    macro_state: MacroState
    liquidity_state: LiquidityState
    volatility_state: VolatilityState
    dealer_state: DealerState
    earnings_state: EarningsState
    time_regime_state: TimeRegimeState
    price_location_state: PriceLocationState
    macro_news_state: MacroNewsState

    # For context
    current_price: Optional[float] = None
    session_open: Optional[float] = None
    session_high: Optional[float] = None
    session_low: Optional[float] = None

    # Session & Flow Context (v1.1.1) - from MarketStateBuilder
    session_name: str = ""  # GLOBEX, PREMARKET, RTH_OPEN, MIDDAY, POWER_HOUR, CLOSE
    session_vol_scale: float = 1.0  # Session-specific volatility scale
    session_liq_scale: float = 1.0  # Session-specific liquidity scale
    session_risk_scale: float = 1.0  # Session-specific risk scale
    prior_high: Optional[float] = None  # Prior day high
    prior_low: Optional[float] = None  # Prior day low
    overnight_high: Optional[float] = None  # Overnight high
    overnight_low: Optional[float] = None  # Overnight low
    vwap: Optional[float] = None  # Volume-weighted average price
    vwap_distance_pct: float = 0.0  # Distance from VWAP as %
    round_level_proximity: Optional[str] = None  # "5000" or "18000" if near round level
    stop_run_detected: bool = False  # True if stop-run pattern detected
    initiative_move_detected: bool = False  # True if initiative move detected
    level_reaction_score: float = 0.0  # -1.0 to 1.0, reaction to key levels


# =============================================================================
# EVALUATION OUTPUT
# =============================================================================


@dataclass
class ScoringFactor:
    """Individual scoring factor result."""

    factor_name: str
    score: float  # [-1, 1]
    weight: float  # [0, 1] in final calculation
    explanation: str
    sub_factors: List[Dict] = field(default_factory=list)  # Detailed breakdown


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    eval_score: float  # [-1, +1] main evaluation
    confidence: float  # [0, 1] confidence in evaluation
    timestamp: datetime
    symbol: str

    # Scoring breakdown
    scoring_factors: List[ScoringFactor]

    # Session & Flow Context (v1.1.1)
    session_name: str = ""
    session_modifiers: Optional[Dict[str, float]] = (
        None  # vol_scale, liq_scale, risk_scale
    )
    flow_signals: Dict[str, Any] = field(
        default_factory=dict
    )  # stop_run, initiative, vwap_dist, etc
    level_reaction_score: float = 0.0
    stop_run_detected: bool = False
    initiative_move_detected: bool = False

    # Regime Intelligence (v2.1)
    regime_label: str = ""  # TREND, RANGE, REVERSAL
    regime_confidence: float = 0.0  # [0, 1] confidence in regime classification
    regime_features: Dict[str, Any] = field(
        default_factory=dict
    )  # Feature breakdown from regime classifier
    regime_adjusted_eval: float = 0.0  # eval_score after regime conditioning
    regime_adjustments: Dict[str, float] = field(
        default_factory=dict
    )  # What was adjusted and by how much

    # Scenario Integration (v2.2)
    scenario_result: Optional[Any] = (
        None  # ScenarioResult with regime-weighted scenarios
    )
    scenario_ev: float = 0.0  # Expected value from scenarios
    scenario_confidence_boost: float = (
        0.0  # Confidence adjustment from scenario alignment
    )

    # Consolidated output
    result_dict: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert -1.0 <= self.eval_score <= 1.0
        assert 0.0 <= self.confidence <= 1.0

        self.result_dict = {
            "eval": round(self.eval_score, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "session": self.session_name,
            "session_modifiers": self.session_modifiers or {},
            "flow_signals": self.flow_signals,
            "regime_label": self.regime_label,
            "regime_confidence": round(self.regime_confidence, 4),
            "regime_adjusted_eval": round(self.regime_adjusted_eval, 4),
            "regime_adjustments": self.regime_adjustments,
            "reasoning": [
                {
                    "factor": sf.factor_name,
                    "score": round(sf.score, 4),
                    "weight": round(sf.weight, 4),
                    "explanation": sf.explanation,
                    "sub_factors": sf.sub_factors,
                }
                for sf in self.scoring_factors
            ],
        }


# =============================================================================
# DEFAULT WEIGHTS
# =============================================================================

DEFAULT_WEIGHTS = {
    "macro": 0.15,
    "liquidity": 0.12,
    "volatility": 0.10,
    "dealer": 0.18,
    "earnings": 0.08,
    "time_regime": 0.10,
    "price_location": 0.12,
    "macro_news": 0.15,
}

# Verify weights sum to 1.0
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


# =============================================================================
# CAUSAL EVALUATOR
# =============================================================================


class CausalEvaluator:
    """
    Stockfish-style market evaluator combining 8 market state factors
    into a unified [-1, +1] evaluation score.

    Each scoring function implements causal market mechanics:
    - Macro: sentiment direction bias
    - Liquidity: absorption/exhaustion dynamics
    - Volatility: expansion/compression behavior
    - Dealer: gamma profile effects
    - Earnings: sector dominance implications
    - Time: session-specific directional risk
    - Price: location-based mean reversion
    - News/Macro: event-driven sentiment

    Output: Deterministic, explainable, production-ready evaluation.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = False,
        official_mode: bool = False,
        enable_regime_conditioning: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            weights: Dict of factor weights. Must sum to 1.0.
            verbose: Enable detailed logging.
            official_mode: Enforce official tournament constraints.
            enable_regime_conditioning: Enable regime-based weight adjustments (v2.1)
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.verbose = verbose
        self.official_mode = official_mode
        self.enable_regime_conditioning = enable_regime_conditioning

        if self.official_mode:
            enforce_official_env(True)
            assert_causal_required(True)
            validate_official_mode_startup(context="causal_evaluator", use_causal=True)

        # Initialize regime classifier if available
        if self.enable_regime_conditioning and REGIME_AVAILABLE:
            self.regime_classifier = RegimeClassifier(logger=logger)
        else:
            self.regime_classifier = None

        # Initialize scenario simulator (v2.2)
        if SCENARIO_AVAILABLE:
            self.scenario_simulator = ScenarioSimulator(verbose=verbose)
        else:
            self.scenario_simulator = None

        # Verify weights
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        self._log(f"[CAUSAL_EVAL] Initialized with weights: {self.weights}")
        if self.regime_classifier:
            self._log(f"[CAUSAL_EVAL] Regime conditioning ENABLED")

    def evaluate(self, state: MarketState) -> EvaluationResult:
        """
        Evaluate market state and return comprehensive result.

        Args:
            state: Complete MarketState with all 8 components

        Returns:
            EvaluationResult with eval_score, confidence, and reasoning

        Raises:
            ValueError: If state is invalid or official_mode constraints violated
        """
        if self.official_mode:
            self._validate_state(state)

        self._log(f"[CAUSAL_EVAL] Evaluating {state.symbol} @ {state.timestamp}")

        # Score each factor
        factors = []

        macro_factor = self._score_macro(state.macro_state)
        factors.append(macro_factor)

        liquidity_factor = self._score_liquidity(state.liquidity_state)
        factors.append(liquidity_factor)

        volatility_factor = self._score_volatility(state.volatility_state)
        factors.append(volatility_factor)

        dealer_factor = self._score_dealer(state.dealer_state)
        factors.append(dealer_factor)

        earnings_factor = self._score_earnings(state.earnings_state)
        factors.append(earnings_factor)

        time_regime_factor = self._score_time_regime(state.time_regime_state)
        factors.append(time_regime_factor)

        price_location_factor = self._score_price_location(state.price_location_state)
        factors.append(price_location_factor)

        macro_news_factor = self._score_macro_news(state.macro_news_state)
        factors.append(macro_news_factor)

        # Combine scores
        eval_score = self._combine_scores(factors)

        # Compute confidence
        confidence = self._compute_confidence(factors, state)

        # Apply session-aware adjustments
        eval_score, confidence = self._apply_session_adjustments(
            eval_score, confidence, state
        )

        # Build flow signals dict
        flow_signals = {
            "vwap": state.vwap,
            "vwap_distance_pct": round(state.vwap_distance_pct, 2),
            "round_level_proximity": state.round_level_proximity,
            "stop_run_detected": state.stop_run_detected,
            "initiative_move_detected": state.initiative_move_detected,
            "prior_high": state.prior_high,
            "prior_low": state.prior_low,
            "overnight_high": state.overnight_high,
            "overnight_low": state.overnight_low,
        }

        # Session modifiers dict
        session_modifiers = {
            "vol_scale": round(state.session_vol_scale, 2),
            "liq_scale": round(state.session_liq_scale, 2),
            "risk_scale": round(state.session_risk_scale, 2),
        }

        # Build result
        result = EvaluationResult(
            eval_score=eval_score,
            confidence=confidence,
            timestamp=state.timestamp,
            symbol=state.symbol,
            scoring_factors=factors,
            session_name=state.session_name,
            session_modifiers=session_modifiers,
            flow_signals=flow_signals,
            level_reaction_score=state.level_reaction_score,
            stop_run_detected=state.stop_run_detected,
            initiative_move_detected=state.initiative_move_detected,
        )

        # Apply regime conditioning (v2.1)
        if self.enable_regime_conditioning and self.regime_classifier:
            result = self._apply_regime_conditioning(result, state)

        # Generate regime-conditioned scenarios (v2.2)
        if (
            self.scenario_simulator
            and state.vwap
            and state.session_high
            and state.session_low
        ):
            result = self._apply_scenario_integration(result, state)

        self._log(
            f"[CAUSAL_EVAL] Result: eval={eval_score:.4f}, conf={confidence:.4f}, session={state.session_name}"
        )

        return result

    # =========================================================================
    # SCORING FUNCTIONS (one per market factor)
    # =========================================================================

    def _score_macro(self, macro_state: MacroState) -> ScoringFactor:
        """
        Score macro environment.

        Causal logic:
        - Hawkish acceleration → negative bias (rates up = less growth)
        - Dovish acceleration → positive bias (rates down = more growth)
        - Beat surprises → positive (economy strong)
        - Miss surprises → negative (economy weak)
        """
        # Sentiment direction (primary)
        sentiment_score = macro_state.sentiment_score

        # Surprise direction (secondary)
        surprise_contribution = macro_state.surprise_score * 0.3

        # Rate expectations
        rate_bias = -macro_state.rate_expectation * 0.4  # Hikes = negative

        # Combine
        combined_score = sentiment_score * 0.5 + surprise_contribution + rate_bias

        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"Macro: sentiment={sentiment_score:.2f}, surprise={macro_state.surprise_score:.2f}, rates={macro_state.rate_expectation:.2f}"

        return ScoringFactor(
            factor_name="Macro",
            score=final_score,
            weight=self.weights["macro"],
            explanation=explanation,
            sub_factors=[
                {"name": "sentiment", "score": sentiment_score},
                {"name": "surprise", "score": macro_state.surprise_score},
                {"name": "rate_expectation", "score": macro_state.rate_expectation},
            ],
        )

    def _score_liquidity(self, liquidity_state: LiquidityState) -> ScoringFactor:
        """
        Score liquidity conditions.

        Causal logic:
        - Absorbing regime → continuation bias (positive, prices absorb orders)
        - Exhausting regime → reversal bias (negative, exhausted buying)
        - Tight spreads → normal continuation
        - Wide spreads → difficulty in moving, slight reversal bias
        """
        regime_score = liquidity_state.regime.value

        # Spread effect: wide = reversal (negative), tight = continuation
        spread_penalty = -min(liquidity_state.bid_ask_spread / 100.0, 0.5)

        # Volume trend: expansion = continuation
        volume_contribution = liquidity_state.volume_trend * 0.3

        # Order book depth
        depth_contribution = liquidity_state.order_book_depth * 0.3

        combined_score = (
            regime_score * 0.5
            + spread_penalty
            + volume_contribution
            + depth_contribution
        )

        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"Liquidity: regime={liquidity_state.regime.name}, spread={liquidity_state.bid_ask_spread:.2f}bp"

        return ScoringFactor(
            factor_name="Liquidity",
            score=final_score,
            weight=self.weights["liquidity"],
            explanation=explanation,
            sub_factors=[
                {"name": "regime", "score": regime_score},
                {"name": "spread_penalty", "score": spread_penalty},
            ],
        )

    def _score_volatility(self, volatility_state: VolatilityState) -> ScoringFactor:
        """
        Score volatility conditions.

        Causal logic:
        - Expanding vol → trend continuation (buyers absorb volatility)
        - Compressing vol → breakout risk (either direction)
        - High vol percentile → extreme conditions, reversal risk
        - Upside skew → bullish (more upside moves), negative skew bearish
        """
        regime_score = volatility_state.regime.value

        # Vol percentile: very high or very low = compression risk
        vol_extremity = abs(volatility_state.vol_percentile - 0.5) * 2  # [0, 1]
        compression_risk = -vol_extremity * 0.3  # Extremes = reversal risk

        # Vol trend
        vol_trend_contribution = volatility_state.vol_trend * 0.3

        # Skew (upside skew = bullish)
        skew_contribution = volatility_state.skew * 0.2

        combined_score = (
            regime_score * 0.5
            + compression_risk
            + vol_trend_contribution
            + skew_contribution
        )

        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"Volatility: regime={volatility_state.regime.name}, percentile={volatility_state.vol_percentile:.2f}, skew={volatility_state.skew:.2f}"

        return ScoringFactor(
            factor_name="Volatility",
            score=final_score,
            weight=self.weights["volatility"],
            explanation=explanation,
            sub_factors=[
                {"name": "regime", "score": regime_score},
                {"name": "percentile", "score": volatility_state.vol_percentile},
                {"name": "skew", "score": volatility_state.skew},
            ],
        )

    def _score_dealer(self, dealer_state: DealerState) -> ScoringFactor:
        """
        Score dealer positioning.

        Causal logic:
        - Positive gamma → market makers are long gamma, mean revert bias (negative)
        - Negative gamma → market makers are short gamma, trend continuation (positive)
        - Long spot exposure → biased upward
        - Short vega → want volatility to contract, slight bullish bias (vol crush helps)
        """
        gamma_effect = (
            -dealer_state.net_gamma_exposure * 0.4
        )  # Positive gamma = mean revert

        spot_effect = dealer_state.net_spot_exposure * 0.3

        vega_effect = (
            -dealer_state.vega_exposure * 0.2
        )  # Short vega wants vol crush (slight up)

        sentiment_effect = dealer_state.dealer_sentiment * 0.1

        combined_score = gamma_effect + spot_effect + vega_effect + sentiment_effect

        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"Dealer: gamma={dealer_state.net_gamma_exposure:.2f}, spot={dealer_state.net_spot_exposure:.2f}, vega={dealer_state.vega_exposure:.2f}"

        return ScoringFactor(
            factor_name="Dealer",
            score=final_score,
            weight=self.weights["dealer"],
            explanation=explanation,
            sub_factors=[
                {"name": "gamma_effect", "score": gamma_effect},
                {"name": "spot_effect", "score": spot_effect},
                {"name": "vega_effect", "score": vega_effect},
            ],
        )

    def _score_earnings(self, earnings_state: EarningsState) -> ScoringFactor:
        """
        Score earnings exposure.

        Causal logic:
        - High mega-cap exposure (NQ dominance) → vol-weighted upside risk
        - Small-cap exposure → more stable but lower upside
        - In earnings season → higher volatility and event risk
        - Beat momentum → positive, miss momentum → negative
        """
        # Mega-cap dominance: higher vol, higher upside
        mega_cap_effect = earnings_state.multi_mega_cap_exposure * 0.3

        # Small-cap drag (inverse effect)
        small_cap_effect = earnings_state.small_cap_exposure * -0.1

        # Earnings season penalty (higher vol = reversal risk)
        earnings_season_penalty = -0.2 if earnings_state.earnings_season_flag else 0.0

        # Beat/miss momentum
        momentum_effect = earnings_state.earnings_surprise_momentum * 0.3

        combined_score = (
            mega_cap_effect
            + small_cap_effect
            + earnings_season_penalty
            + momentum_effect
        )

        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"Earnings: mega_cap={earnings_state.multi_mega_cap_exposure:.2f}, earnings_season={earnings_state.earnings_season_flag}, momentum={earnings_state.earnings_surprise_momentum:.2f}"

        return ScoringFactor(
            factor_name="Earnings",
            score=final_score,
            weight=self.weights["earnings"],
            explanation=explanation,
            sub_factors=[
                {"name": "mega_cap_exposure", "score": mega_cap_effect},
                {"name": "earnings_season_penalty", "score": earnings_season_penalty},
            ],
        )

    def _score_time_regime(self, time_regime_state: TimeRegimeState) -> ScoringFactor:
        """
        Score time of day and session effects.

        Causal logic:
        - NY Open → highest volatility, highest trend strength (both directions possible)
        - Power Hour (last hour) → trend continuation strongest
        - Asian hours → lower volatility, range-bound
        - London Close → transition volume
        """
        base_score = time_regime_state.regime_type.value

        # Session progression effect (tends toward close)
        session_progress = (
            time_regime_state.minutes_into_session / 960.0
        )  # 16 hours = 960 min
        progress_effect = (session_progress - 0.5) * 0.2  # Slight negative late session

        # Day of week: Monday = reverse, Friday = close early
        day_of_week_score = 0.0
        if time_regime_state.day_of_week == 0:  # Monday
            day_of_week_score = -0.1  # Reversals common after weekend
        elif time_regime_state.day_of_week == 4:  # Friday
            day_of_week_score = -0.05  # Position squaring

        combined_score = base_score + progress_effect + day_of_week_score
        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"Time: regime={time_regime_state.regime_type.name}, day={time_regime_state.day_of_week}"

        return ScoringFactor(
            factor_name="TimeRegime",
            score=final_score,
            weight=self.weights["time_regime"],
            explanation=explanation,
            sub_factors=[
                {"name": "regime_base", "score": base_score},
                {"name": "session_progress", "score": progress_effect},
            ],
        )

    def _score_price_location(
        self, price_location_state: PriceLocationState
    ) -> ScoringFactor:
        """
        Score price location effects.

        Causal logic:
        - Near session highs → reversal risk (short term mean reversion)
        - Near session lows → reversal risk upward
        - Mid-range → neutral, no edge
        - Extremity % used as reversal strength
        """
        session_extremity = price_location_state.session_extremity

        # Extremity = mean reversion risk (both directions possible)
        # At high = reversal down bias, at low = reversal up bias
        extremity_magnitude = abs(session_extremity)
        reversal_bias = -session_extremity  # Flip direction for mean reversion
        reversal_strength = extremity_magnitude * 0.5

        mean_reversion_score = reversal_bias * reversal_strength

        # Range ratio: expanded range = volatility, contraction = breakout
        range_effect = (price_location_state.range_ratio - 1.0) * 0.2

        combined_score = mean_reversion_score + range_effect
        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"PriceLocation: extremity={session_extremity:.2f}, range_ratio={price_location_state.range_ratio:.2f}"

        return ScoringFactor(
            factor_name="PriceLocation",
            score=final_score,
            weight=self.weights["price_location"],
            explanation=explanation,
            sub_factors=[
                {"name": "mean_reversion_bias", "score": mean_reversion_score},
                {"name": "range_effect", "score": range_effect},
            ],
        )

    def _score_macro_news(self, macro_news_state: MacroNewsState) -> ScoringFactor:
        """
        Score macro news and events.

        Causal logic:
        - Risk-on sentiment → positive bias
        - Risk-off sentiment → negative bias
        - High event importance → weight multiplier
        - Recent events → higher confidence
        - Hawkish surprise → mixed (good economy but higher rates)
        """
        base_risk_sentiment = macro_news_state.risk_sentiment_score

        # Hawkishness has mixed effect: bad for fixed income, good for economy
        hawkishness_effect = macro_news_state.hawkishness_score * 0.1

        # Surprise: beats = positive
        surprise_effect = macro_news_state.surprise_score * 0.2

        # Event importance as weight multiplier
        importance_weight = 0.5 + (macro_news_state.event_importance / 3.0) * 0.5

        # Recency: recent events = higher conviction
        recency_factor = min(1.0, 1.0 / (macro_news_state.hours_since_last_event + 1.0))

        # Event frequency: multiple events = confirmation
        frequency_factor = min(1.0, macro_news_state.macro_event_count / 5.0)

        combined_score = (
            base_risk_sentiment * importance_weight * recency_factor
            + hawkishness_effect
            + surprise_effect
            + frequency_factor * 0.1
        )

        final_score = np.clip(combined_score, -1.0, 1.0)

        explanation = f"MacroNews: risk_sentiment={base_risk_sentiment:.2f}, importance={macro_news_state.event_importance}, state={macro_news_state.macro_news_state}"

        return ScoringFactor(
            factor_name="MacroNews",
            score=final_score,
            weight=self.weights["macro_news"],
            explanation=explanation,
            sub_factors=[
                {"name": "risk_sentiment", "score": base_risk_sentiment},
                {"name": "hawkishness", "score": hawkishness_effect},
                {"name": "surprise", "score": surprise_effect},
                {"name": "importance", "value": macro_news_state.event_importance},
            ],
        )

    # =========================================================================
    # COMBINATION & CONFIDENCE
    # =========================================================================

    def _combine_scores(self, factors: List[ScoringFactor]) -> float:
        """
        Combine individual factor scores using weighted linear combination.

        EvalScore = Σ (weight_i * score_i)

        Args:
            factors: List of ScoringFactor objects

        Returns:
            Combined score in [-1, +1]
        """
        combined = sum(factor.weight * factor.score for factor in factors)
        return np.clip(combined, -1.0, 1.0)

    def _compute_confidence(
        self, factors: List[ScoringFactor], state: MarketState
    ) -> float:
        """
        Compute confidence in the evaluation.

        Confidence is based on:
        - Agreement between factors (low variance = high confidence)
        - Volatility regime clarity
        - Liquidity conditions
        - Macro event recency

        Returns:
            Confidence in [0, 1]
        """
        scores = np.array([f.score for f in factors])

        # Agreement: low variance = high confidence
        score_variance = np.var(scores)
        agreement_confidence = 1.0 / (1.0 + score_variance)  # [0, 1]

        # Liquidity: better liquidity = higher confidence
        bid_ask = state.liquidity_state.bid_ask_spread
        liquidity_confidence = 1.0 / (1.0 + bid_ask / 10.0)  # [0, 1]

        # Volatility: moderate vol = higher confidence
        vol_percentile = state.volatility_state.vol_percentile
        vol_extremity = abs(vol_percentile - 0.5) * 2  # [0, 1]
        volatility_confidence = (
            1.0 - vol_extremity * 0.3
        )  # Extreme vol = lower confidence

        # Macro: recent event = higher confidence
        hours_since_event = state.macro_news_state.hours_since_last_event
        macro_confidence = 1.0 / (1.0 + hours_since_event / 24.0)  # Decays over 24h

        # Combine confidences (equal weight)
        final_confidence = (
            agreement_confidence * 0.3
            + liquidity_confidence * 0.2
            + volatility_confidence * 0.2
            + macro_confidence * 0.3
        )

        return np.clip(final_confidence, 0.0, 1.0)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _validate_state(self, state: MarketState) -> None:
        """Validate state in official tournament mode."""
        if not isinstance(state, MarketState):
            raise ValueError("State must be MarketState object")

        if state.timestamp > datetime.now():
            raise ValueError(
                f"[OFFICIAL_TOURNAMENT] Cannot evaluate future timestamp: {state.timestamp}"
            )

        # Check all components present
        required_components = [
            "macro_state",
            "liquidity_state",
            "volatility_state",
            "dealer_state",
            "earnings_state",
            "time_regime_state",
            "price_location_state",
            "macro_news_state",
        ]

        for component in required_components:
            if not hasattr(state, component):
                raise ValueError(
                    f"[OFFICIAL_TOURNAMENT] Missing component: {component}"
                )

    def _apply_session_adjustments(
        self, eval_score: float, confidence: float, state: MarketState
    ) -> tuple:
        """
        Apply session-aware weighting adjustments to evaluation score and confidence.

        Session-specific strategies:
        - GLOBEX (overnight): reduce confidence (low liquidity, overnight gaps)
        - PREMARKET: increase mean-reversion weight (overnight extremes)
        - RTH_OPEN: increase uncertainty penalty (volume ramp-up)
        - MIDDAY: increase mean-reversion weight (mean reversion strongest at midday)
        - POWER_HOUR: increase trend weight (flow bias strongest)
        - CLOSE: flow persistence, increase recent trend weight

        Args:
            eval_score: Base evaluation score [-1, 1]
            confidence: Base confidence [0, 1]
            state: MarketState with session context

        Returns:
            Tuple of (adjusted_eval_score, adjusted_confidence)
        """
        if not hasattr(state, "session_name") or not state.session_name:
            return eval_score, confidence

        session = state.session_name
        vol_scale = state.session_vol_scale
        liq_scale = state.session_liq_scale
        risk_scale = state.session_risk_scale

        # Apply flow-aware adjustments
        adjustment = 0.0
        confidence_adj = 0.0

        if session == "GLOBEX":
            # Overnight: lower confidence, prefer mean-reversion
            confidence_adj = -0.15 * liq_scale
            if abs(eval_score) < 0.3:
                adjustment = -eval_score * 0.05  # slight mean-reversion bias

        elif session == "PREMARKET":
            # Pre-open: high uncertainty, overnight extremes attract reversal
            confidence_adj = -0.10
            if abs(eval_score) > 0.5:
                adjustment = (
                    -eval_score * 0.1
                )  # encourage mean-reversion of overnight extremes

        elif session == "RTH_OPEN":
            # Open: volume ramp-up, uncertainty increases
            confidence_adj = -0.05
            # Stop-run detection: penalize strong signals at open
            if state.stop_run_detected:
                adjustment = -eval_score * 0.15

        elif session == "MIDDAY":
            # Midday: mean-reversion typically strongest, highest data quality
            confidence_adj = 0.05
            # Boost mean-reversion component if applicable
            if abs(eval_score) > 0.3 and state.vwap_distance_pct > 1.0:
                adjustment = -eval_score * 0.08  # slight mean-reversion to VWAP

        elif session == "POWER_HOUR":
            # 3-4 PM: flow-driven, trend continuation is stronger
            confidence_adj = 0.05
            # Boost trend component
            if state.initiative_move_detected:
                adjustment = eval_score * 0.10  # reward initiative moves
            if (
                state.round_level_proximity is not None
                and state.round_level_proximity > 0.8
            ):
                adjustment = eval_score * 0.08  # near levels support trend

        elif session == "CLOSE":
            # Close: flow persistence, fund activity
            confidence_adj = 0.00
            # Trend continuation from POWER_HOUR
            if abs(eval_score) > 0.3:
                adjustment = eval_score * 0.05

        # Apply level reaction adjustments (cross-session)
        if state.level_reaction_score > 0.5 and abs(eval_score) > 0.4:
            # Strong level reaction at extremes suggests quality signal
            confidence_adj += 0.05

        # Apply stop-run penalty (cross-session)
        if state.stop_run_detected:
            confidence_adj -= 0.03  # slight hit to confidence

        # Compute adjusted score and confidence
        adjusted_score = np.clip(eval_score + adjustment, -1.0, 1.0)
        adjusted_confidence = np.clip(confidence + confidence_adj, 0.0, 1.0)

        return adjusted_score, adjusted_confidence

    def _apply_regime_conditioning(
        self, result: EvaluationResult, state: MarketState
    ) -> EvaluationResult:
        """
        Apply regime-conditioned modifications to evaluation score.

        Regime adaptations (v2.1):

        TREND REGIME:
        - Increase weight on initiative signals
        - Increase trend-following bias
        - Reduce mean-reversion scoring
        - Increase confidence when aligned with direction
        - Allow stronger positions in trend direction

        RANGE REGIME:
        - Increase weight on oscillation/compression signals
        - Increase mean-reversion scoring
        - Reduce trend-following scoring
        - Reduce confidence on breakouts
        - Reduce position sizes overall

        REVERSAL REGIME:
        - Increase weight on failed breakout signals
        - Increase VWAP reversion scoring
        - Increase level-reaction scoring
        - Reduce confidence on continuation moves
        - Tighten risk controls

        Args:
            result: Base EvaluationResult
            state: MarketState with OHLCV data

        Returns:
            Modified EvaluationResult with regime adjustments applied
        """
        try:
            # Update regime classifier with current bar
            regime_state = self.regime_classifier.update_with_bar(
                timestamp=state.timestamp,
                open_price=state.current_price or 0.0,
                high=state.session_high or state.current_price or 0.0,
                low=state.session_low or state.current_price or 0.0,
                close=state.current_price or 0.0,
                volume=1000,  # Dummy volume, not used in regime calc
                vwap=state.vwap or state.current_price or 0.0,
                flow_signals=state.flow_signals or {},
            )

            result.regime_label = regime_state.regime_label
            result.regime_confidence = regime_state.regime_confidence
            result.regime_features = regime_state.regime_features

            # Apply regime-specific adjustments
            if regime_state.regime_label == "TREND":
                result = self._condition_for_trend(result, regime_state)
            elif regime_state.regime_label == "RANGE":
                result = self._condition_for_range(result, regime_state)
            elif regime_state.regime_label == "REVERSAL":
                result = self._condition_for_reversal(result, regime_state)

        except Exception as e:
            self._log(f"[REGIME_CONDITIONING] Error: {e}, skipping regime adjustments")
            # Fall back to base result
            result.regime_label = "UNKNOWN"
            result.regime_confidence = 0.0

        return result

    def _condition_for_trend(
        self, result: EvaluationResult, regime_state: RegimeState
    ) -> EvaluationResult:
        """Apply TREND regime conditioning to evaluation."""
        # Boost eval_score if trending in right direction
        if abs(result.eval_score) > 0.3:
            # Strengthen trend signal
            boost = result.eval_score * 0.15 * regime_state.regime_confidence
            result.regime_adjusted_eval = np.clip(result.eval_score + boost, -1.0, 1.0)

            # Increase confidence in trend direction
            conf_boost = 0.10 * regime_state.regime_confidence
            result.confidence = np.clip(result.confidence + conf_boost, 0.0, 1.0)

            result.regime_adjustments = {
                "type": "trend_boost",
                "eval_adjustment": round(boost, 4),
                "confidence_adjustment": round(conf_boost, 4),
            }
        else:
            # Low signal, slightly penalize mean-reversion in TREND
            penalty = abs(result.eval_score) * 0.05
            result.regime_adjusted_eval = result.eval_score
            result.regime_adjustments = {
                "type": "trend_low_signal",
                "note": "Low conviction signal in TREND regime",
            }

        return result

    def _condition_for_range(
        self, result: EvaluationResult, regime_state: RegimeState
    ) -> EvaluationResult:
        """Apply RANGE regime conditioning to evaluation."""
        # In RANGE, favor mean-reversion signals
        if abs(result.eval_score) > 0.5:
            # Strong signals in RANGE are suspicious (false breakouts)
            # Reduce confidence
            conf_penalty = 0.15 * regime_state.regime_confidence
            result.confidence = np.clip(result.confidence - conf_penalty, 0.0, 1.0)

            # Reverse signal (fade breakouts)
            fade_boost = -result.eval_score * 0.08 * regime_state.regime_confidence
            result.regime_adjusted_eval = np.clip(
                result.eval_score + fade_boost, -1.0, 1.0
            )

            result.regime_adjustments = {
                "type": "range_fade_breakout",
                "eval_adjustment": round(fade_boost, 4),
                "confidence_penalty": round(conf_penalty, 4),
            }
        else:
            # Weaker signals favor mean-reversion
            result.regime_adjusted_eval = result.eval_score
            result.regime_adjustments = {
                "type": "range_low_signal",
                "note": "Prefer mean-reversion in RANGE regime",
            }

        return result

    def _condition_for_reversal(
        self, result: EvaluationResult, regime_state: RegimeState
    ) -> EvaluationResult:
        """Apply REVERSAL regime conditioning to evaluation."""
        # In REVERSAL, counter-trend entries after exhaustion
        if abs(result.eval_score) < 0.4:
            # Lower conviction signals in REVERSAL are actually useful (exhaustion)
            conf_boost = 0.12 * regime_state.regime_confidence
            result.confidence = np.clip(result.confidence + conf_boost, 0.0, 1.0)

            # Slightly reverse (prepare for counter-move)
            counter_boost = -result.eval_score * 0.10 * regime_state.regime_confidence
            result.regime_adjusted_eval = np.clip(
                result.eval_score + counter_boost, -1.0, 1.0
            )

            result.regime_adjustments = {
                "type": "reversal_counter_entry",
                "eval_adjustment": round(counter_boost, 4),
                "confidence_boost": round(conf_boost, 4),
            }
        else:
            # Strong signals in REVERSAL suggest we're still in exhaustion phase
            conf_penalty = 0.10 * regime_state.regime_confidence
            result.confidence = np.clip(result.confidence - conf_penalty, 0.0, 1.0)

            result.regime_adjustments = {
                "type": "reversal_exhaustion_phase",
                "confidence_penalty": round(conf_penalty, 4),
                "note": "Strong signal suggests exhaustion not yet complete",
            }

        return result

    def _apply_scenario_integration(
        self, result: EvaluationResult, state: MarketState
    ) -> EvaluationResult:
        """
        Generate regime-conditioned scenarios and integrate into evaluation.

        Scenarios provide an internal "search tree" that reflects the regime:
        - TREND: Continuation paths weighted heavily
        - RANGE: Oscillation paths weighted equally
        - REVERSAL: Reversal paths weighted heavily

        The scenario expected value and alignment adjust the final confidence.
        """

        # Defensive numeric extraction (mocks may return non-numeric values)
        def _safe_float(val: Any) -> Optional[float]:
            try:
                converted = float(val)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(converted):
                return None
            return converted

        session_high = _safe_float(getattr(state, "session_high", None))
        session_low = _safe_float(getattr(state, "session_low", None))
        current_price = _safe_float(getattr(state, "current_price", None))
        vwap = _safe_float(getattr(state, "vwap", None))

        # Calculate expected move from volatility; fall back to neutral if invalid
        if session_high is not None and session_low is not None:
            range_val = session_high - session_low
            expected_move = range_val * 0.6  # Typically 60% of session range
        elif current_price is not None:
            expected_move = current_price * 0.01
        else:
            expected_move = 0.01

        # Generate scenarios with regime conditioning
        scenario_result = self.scenario_simulator.simulate_scenarios(
            current_price=current_price or 0.0,
            vwap=vwap if vwap is not None else (current_price or 0.0),
            session_high=session_high or 0.0,
            session_low=session_low or 0.0,
            expected_move=expected_move,
            volatility=state.volatility_state.current_vol,
            regime_label=result.regime_label,
            regime_confidence=result.regime_confidence,
            eval_score=result.eval_score,
        )

        # Store scenario result
        result.scenario_result = scenario_result

        # Calculate scenario EV as proportion of current price
        if current_price and current_price > 0:
            result.scenario_ev = (
                scenario_result.expected_price - current_price
            ) / current_price

        # Adjust confidence based on scenario alignment
        # High alignment → confidence boost, low alignment → confidence penalty
        confidence_adjustment = (
            scenario_result.regime_alignment * 0.10 * result.regime_confidence
        )
        result.scenario_confidence_boost = confidence_adjustment
        result.confidence = np.clip(result.confidence + confidence_adjustment, 0.0, 1.0)

        self._log(
            f"[CAUSAL_EVAL] Scenario integration: bias={scenario_result.scenario_bias}, "
            f"alignment={scenario_result.regime_alignment:.2f}, conf_boost={confidence_adjustment:.4f}"
        )

        return result

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            logger.info(message)

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update evaluation weights.

        Args:
            weights: Dict of factor weights. Must sum to 1.0.
        """
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        self.weights = weights.copy()
        self._log(f"[CAUSAL_EVAL] Updated weights: {self.weights}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_causal_evaluator(
    custom_weights: Optional[Dict[str, float]] = None,
    official_mode: bool = False,
    verbose: bool = False,
) -> CausalEvaluator:
    """
    Factory function to create a CausalEvaluator.

    Args:
        custom_weights: Optional custom weights (must sum to 1.0)
        official_mode: Enable official tournament constraints
        verbose: Enable logging

    Returns:
        CausalEvaluator instance
    """
    return CausalEvaluator(
        weights=custom_weights, official_mode=official_mode, verbose=verbose
    )


# ============================================================================
# REGIME-CONDITIONED, DETERMINISTIC EVALUATION (v5.0-A)
# ============================================================================


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_factor(score: float) -> float:
    return clamp(score, -1.0, 1.0)


def combine_confidence(components: List[float]) -> float:
    if not components:
        return 0.0
    avg = sum(abs(c) for c in components) / len(components)
    return clamp(avg, 0.0, 1.0)


def _vol_factor(state: SchemaMarketState, regime: RegimeSignal) -> Dict[str, Any]:
    raw_score = normalize_factor(0.5 - float(state.volatility.realized_vol))
    weight = FACTOR_WEIGHTS["VOL"].get(regime.vol, 0.3)
    weighted = clamp(raw_score * weight, -1.0, 1.0)
    return {"raw": raw_score, "weight": weight, "weighted": weighted}


def _liq_factor(state: SchemaMarketState, regime: RegimeSignal) -> Dict[str, Any]:
    depth_total = float(
        state.liquidity.cumulative_depth_bid + state.liquidity.cumulative_depth_ask
    )
    pressure = float(state.liquidity.liquidity_pressure)
    raw_score = normalize_factor(depth_total / 1_000_000.0 - pressure * 0.1)
    weight = FACTOR_WEIGHTS["LIQ"].get(regime.liq, 0.3)
    weighted = clamp(raw_score * weight, -1.0, 1.0)
    return {"raw": raw_score, "weight": weight, "weighted": weighted}


def _macro_factor(state: SchemaMarketState, regime: RegimeSignal) -> Dict[str, Any]:
    rs = float(state.macro.risk_sentiment)
    hawk = float(state.macro.hawkishness)
    surprise = float(state.macro.surprise_score)
    raw_score = normalize_factor(rs + surprise * 0.2 - hawk * 0.1)
    weight = FACTOR_WEIGHTS["MACRO"].get(regime.macro, 0.25)
    weighted = clamp(raw_score * weight, -1.0, 1.0)
    return {"raw": raw_score, "weight": weight, "weighted": weighted}


def evaluate_state(state: SchemaMarketState) -> Dict[str, Any]:
    """Deterministic regime-conditioned evaluation (v5.0-A).

    Args:
        state: Canonical MarketState (immutable, serializable)

    Returns:
        JSON-serializable dict with score, confidence, factor breakdown, regime metadata.
    """

    if not isinstance(state, SchemaMarketState):
        raise TypeError("state must be state.schema.MarketState")

    reg_raw = state.raw.get("regime") if isinstance(state.raw, dict) else None
    if reg_raw is None:
        raise ValueError(
            "state.raw['regime'] missing for regime-conditioned evaluation"
        )
    if isinstance(reg_raw, RegimeSignal):
        regime = reg_raw
    elif isinstance(reg_raw, dict):
        regime = RegimeSignal(**reg_raw)
    else:
        raise TypeError("state.raw['regime'] must be RegimeSignal or dict")

    vol = _vol_factor(state, regime)
    liq = _liq_factor(state, regime)
    macro = _macro_factor(state, regime)

    swing_structure = getattr(regime, "swing_structure", None) or getattr(
        state, "swing_structure", "NEUTRAL"
    )
    trend_direction = getattr(regime, "trend_direction", None) or getattr(
        state, "trend_direction", "RANGE"
    )
    trend_strength = float(
        getattr(regime, "trend_strength", 0.0)
        or getattr(state, "trend_strength", 0.0)
        or 0.0
    )
    swing_high = float(
        getattr(regime, "swing_high", 0.0) or getattr(state, "swing_high", 0.0) or 0.0
    )
    swing_low = float(
        getattr(regime, "swing_low", 0.0) or getattr(state, "swing_low", 0.0) or 0.0
    )

    final_score = clamp(
        vol["weighted"] + liq["weighted"] + macro["weighted"], -1.0, 1.0
    )
    confidence = combine_confidence([vol["raw"], liq["raw"], macro["raw"]])

    vol_shock = bool(getattr(state.volatility, "volatility_shock", False))
    vol_shock_strength = float(
        getattr(state.volatility, "volatility_shock_strength", 0.0)
    )
    if vol_shock:
        final_score = clamp(
            final_score * max(0.4, 1.0 - 0.6 * vol_shock_strength), -1.0, 1.0
        )
        confidence = clamp(
            confidence * max(0.5, 1.0 - 0.5 * vol_shock_strength), 0.0, 1.0
        )

    result = {
        "score": final_score,
        "confidence": confidence,
        "factors": {
            "volatility": vol,
            "liquidity": liq,
            "macro": macro,
        },
        "regime": regime.to_dict(),
        "version": "v5.0-A",
    }
    result["trend"] = {
        "swing_structure": swing_structure,
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }
    if vol_shock:
        result["volatility_shock"] = True
        result["volatility_shock_strength"] = vol_shock_strength
    return result


def get_default_market_state(
    symbol: str = "EUR/USD", timestamp: Optional[datetime] = None
) -> MarketState:
    """
    Create a neutral default MarketState for testing.

    All factors set to neutral values.
    """
    if timestamp is None:
        timestamp = datetime.now()

    return MarketState(
        timestamp=timestamp,
        symbol=symbol,
        macro_state=MacroState(
            sentiment_score=0.0,
            surprise_score=0.0,
            rate_expectation=0.0,
            inflation_expectation=0.0,
            gdp_expectation=0.0,
        ),
        liquidity_state=LiquidityState(
            bid_ask_spread=2.0,
            order_book_depth=0.5,
            regime=LiquidityRegime.NORMAL,
            volume_trend=0.0,
        ),
        volatility_state=VolatilityState(
            current_vol=0.10,
            vol_percentile=0.5,
            regime=VolatilityRegime.NORMAL,
            vol_trend=0.0,
            skew=0.0,
        ),
        dealer_state=DealerState(
            net_gamma_exposure=0.0,
            net_spot_exposure=0.0,
            vega_exposure=0.0,
            dealer_sentiment=0.0,
        ),
        earnings_state=EarningsState(
            multi_mega_cap_exposure=0.5,
            small_cap_exposure=0.5,
            earnings_season_flag=False,
            earnings_surprise_momentum=0.0,
        ),
        time_regime_state=TimeRegimeState(
            regime_type=TimeRegimeType.LONDON_OPEN,
            minutes_into_session=60,
            hours_until_session_end=8,
            day_of_week=2,
        ),
        price_location_state=PriceLocationState(
            distance_from_high=0.5,
            distance_from_low=0.5,
            range_ratio=1.0,
            session_extremity=0.0,
        ),
        macro_news_state=MacroNewsState(
            risk_sentiment_score=0.0,
            hawkishness_score=0.0,
            surprise_score=0.0,
            event_importance=0,
            hours_since_last_event=24.0,
            macro_event_count=0,
            news_article_count=0,
            macro_news_state="NEUTRAL",
        ),
    )
