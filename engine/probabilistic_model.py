"""
Probabilistic Model: Action-Conditioned Return Modeling (Phase v3.0).

Transforms deterministic evaluation scores into risk-adjusted expected value
distributions conditioned on regime, scenario probabilities, volatility,
liquidity, flow signals, and session context.

Core principle: Every action has a probabilistic distribution of outcomes.
This module computes those distributions and their risk-adjusted expectations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import math


class MoveType(Enum):
    """Types of potential market moves."""
    UP = "UP"
    DOWN = "DOWN"
    CHOP = "CHOP"


@dataclass
class MoveExpectation:
    """Expected value and risk metrics for a specific move type."""
    move_type: MoveType
    base_return: float  # Raw expected return before risk adjustment
    regime_multiplier: float  # [0.5, 1.5] regime boost/penalty
    flow_multiplier: float  # [0.7, 1.3] flow context boost/penalty
    volatility_penalty: float  # Reduction due to vol uncertainty
    liquidity_penalty: float  # Reduction due to liquidity constraints
    risk_penalty: float  # Combined risk reduction
    risk_adjusted_return: float  # base_return * multipliers - penalties
    probability: float  # [0, 1] scenario probability of this move
    expected_value: float  # risk_adjusted_return * probability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "move_type": self.move_type.value,
            "base_return": round(self.base_return, 6),
            "regime_multiplier": round(self.regime_multiplier, 4),
            "flow_multiplier": round(self.flow_multiplier, 4),
            "volatility_penalty": round(self.volatility_penalty, 6),
            "liquidity_penalty": round(self.liquidity_penalty, 6),
            "risk_penalty": round(self.risk_penalty, 6),
            "risk_adjusted_return": round(self.risk_adjusted_return, 6),
            "probability": round(self.probability, 4),
            "expected_value": round(self.expected_value, 6),
        }


@dataclass
class ProbabilisticDistribution:
    """Complete probabilistic distribution of outcomes."""
    up_expectation: MoveExpectation
    down_expectation: MoveExpectation
    chop_expectation: MoveExpectation
    
    # Aggregate metrics
    combined_expected_value: float  # Sum of all expected values
    risk_adjusted_ev: float  # EV adjusted for uncertainty premium
    best_move_type: MoveType  # Highest risk-adjusted EV
    best_move_ev: float  # Its expected value
    distribution_confidence: float  # [0, 1] confidence in distribution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "up": self.up_expectation.to_dict(),
            "down": self.down_expectation.to_dict(),
            "chop": self.chop_expectation.to_dict(),
            "combined_expected_value": round(self.combined_expected_value, 6),
            "risk_adjusted_ev": round(self.risk_adjusted_ev, 6),
            "best_move_type": self.best_move_type.value,
            "best_move_ev": round(self.best_move_ev, 6),
            "distribution_confidence": round(self.distribution_confidence, 4),
        }


class ProbabilisticModel:
    """
    Computes risk-adjusted expected value distributions for market moves.
    
    Integrates regime, scenario probabilities, volatility, liquidity, flow signals,
    and session context to generate action-conditioned return distributions.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize probabilistic model."""
        self.verbose = verbose
    
    def compute_distribution(
        self,
        regime_label: str,  # TREND, RANGE, REVERSAL
        regime_confidence: float,  # [0, 1]
        probability_up: float,  # [0, 1]
        probability_down: float,  # [0, 1]
        probability_chop: float,  # [0, 1]
        eval_score: float,  # [-1, +1] base evaluation
        volatility: float,  # [0, 1] current volatility
        liquidity_score: float,  # [0, 1] where 1 = very liquid
        volatility_percentile: float,  # [0, 1] percentile of historical
        stop_run_detected: bool,  # True if stop run detected
        initiative_detected: bool,  # True if initiative move detected
        level_reaction_score: float,  # [-1, +1] reaction strength
        session_name: str = "",  # GLOBEX, RTH_OPEN, MIDDAY, CLOSE, etc.
    ) -> ProbabilisticDistribution:
        """
        Compute risk-adjusted expected value distribution.
        
        Args:
            regime_label: Current market regime
            regime_confidence: Confidence in regime classification
            probability_up: Scenario probability of UP move
            probability_down: Scenario probability of DOWN move
            probability_chop: Scenario probability of CHOP move
            eval_score: Base evaluation score [-1, +1]
            volatility: Current volatility level
            liquidity_score: Market liquidity 0-1
            volatility_percentile: Historical volatility percentile
            stop_run_detected: Whether stop-run pattern detected
            initiative_detected: Whether initiative move detected
            level_reaction_score: Strength of level reactions
            session_name: Current trading session
            
        Returns:
            ProbabilisticDistribution with expected values and risk metrics
        """
        # Compute regime multipliers
        regime_mults = self._compute_regime_multipliers(regime_label, eval_score)
        
        # Compute flow multipliers
        flow_mults = self._compute_flow_multipliers(
            stop_run_detected,
            initiative_detected,
            level_reaction_score,
            session_name
        )
        
        # Compute risk penalties
        vol_penalty = self._compute_volatility_penalty(volatility, volatility_percentile)
        liq_penalty = self._compute_liquidity_penalty(liquidity_score)
        regime_confidence_adjustment = self._compute_regime_confidence_adjustment(regime_confidence)
        total_risk_penalty = vol_penalty + liq_penalty - regime_confidence_adjustment
        total_risk_penalty = max(0.0, min(0.5, total_risk_penalty))  # Clamp to [0, 0.5]
        
        # Compute base expected returns
        base_ev_up = self._compute_base_return_up(volatility, eval_score, probability_up)
        base_ev_down = self._compute_base_return_down(volatility, eval_score, probability_down)
        base_ev_chop = self._compute_base_return_chop(volatility, eval_score, probability_chop)
        
        # Apply multipliers and penalties
        up_exp = self._build_move_expectation(
            MoveType.UP,
            base_ev_up,
            regime_mults["up"],
            flow_mults["up"],
            vol_penalty,
            liq_penalty,
            total_risk_penalty,
            probability_up
        )
        
        down_exp = self._build_move_expectation(
            MoveType.DOWN,
            base_ev_down,
            regime_mults["down"],
            flow_mults["down"],
            vol_penalty,
            liq_penalty,
            total_risk_penalty,
            probability_down
        )
        
        chop_exp = self._build_move_expectation(
            MoveType.CHOP,
            base_ev_chop,
            regime_mults["chop"],
            flow_mults["chop"],
            vol_penalty,
            liq_penalty,
            total_risk_penalty,
            probability_chop
        )
        
        # Compute aggregate metrics
        combined_ev = up_exp.expected_value + down_exp.expected_value + chop_exp.expected_value
        risk_adjusted_ev = combined_ev - total_risk_penalty
        
        # Determine best move
        best_exp, best_type, best_ev = self._find_best_move(up_exp, down_exp, chop_exp)
        
        # Compute distribution confidence
        dist_confidence = self._compute_distribution_confidence(
            regime_confidence,
            max(probability_up, probability_down, probability_chop),
            abs(combined_ev)
        )
        
        return ProbabilisticDistribution(
            up_expectation=up_exp,
            down_expectation=down_exp,
            chop_expectation=chop_exp,
            combined_expected_value=combined_ev,
            risk_adjusted_ev=risk_adjusted_ev,
            best_move_type=best_type,
            best_move_ev=best_ev,
            distribution_confidence=dist_confidence
        )
    
    def _compute_regime_multipliers(self, regime_label: str, eval_score: float) -> Dict[str, float]:
        """Compute regime-specific multipliers for each move type."""
        mults = {"up": 1.0, "down": 1.0, "chop": 1.0}
        
        if regime_label == "TREND":
            # TREND regime favors continuation
            if eval_score > 0.3:  # Bullish eval
                mults["up"] = 1.25  # Boost up moves
                mults["chop"] = 0.85  # Penalize chop
                mults["down"] = 0.90  # Penalize down
            elif eval_score < -0.3:  # Bearish eval
                mults["down"] = 1.25  # Boost down moves
                mults["chop"] = 0.85  # Penalize chop
                mults["up"] = 0.90  # Penalize up
            else:
                # Neutral eval in trend = ambiguous
                mults["up"] = 1.05
                mults["down"] = 1.05
                mults["chop"] = 0.95
        
        elif regime_label == "RANGE":
            # RANGE regime favors mean-reversion and chop
            mults["chop"] = 1.30  # Strongly boost chop
            mults["up"] = 0.85  # Penalize directional moves
            mults["down"] = 0.85
        
        elif regime_label == "REVERSAL":
            # REVERSAL regime penalizes continuation, favors reversal
            if eval_score > 0.3:  # Bullish eval
                mults["down"] = 1.20  # Boost down (reversal)
                mults["up"] = 0.80  # Penalize up
                mults["chop"] = 1.05
            elif eval_score < -0.3:  # Bearish eval
                mults["up"] = 1.20  # Boost up (reversal)
                mults["down"] = 0.80  # Penalize down
                mults["chop"] = 1.05
            else:
                mults["up"] = 1.10
                mults["down"] = 1.10
                mults["chop"] = 1.00
        
        return mults
    
    def _compute_flow_multipliers(
        self,
        stop_run_detected: bool,
        initiative_detected: bool,
        level_reaction_score: float,
        session_name: str
    ) -> Dict[str, float]:
        """Compute flow-context-based multipliers."""
        mults = {"up": 1.0, "down": 1.0, "chop": 1.0}
        
        # Stop-run detection reduces all directional moves
        if stop_run_detected:
            mults["up"] *= 0.75
            mults["down"] *= 0.75
            mults["chop"] *= 1.10
        
        # Initiative move detected
        if initiative_detected:
            if session_name in ["RTH_OPEN", "POWER_HOUR"]:
                # Strong continuation likelihood
                mults["up"] *= 1.15
                mults["down"] *= 1.15
                mults["chop"] *= 0.85
            else:
                # Outside strong sessions, less predictive
                mults["up"] *= 1.05
                mults["down"] *= 1.05
                mults["chop"] *= 1.00
        
        # Level reaction strength
        if level_reaction_score > 0.6:
            # Strong bullish reaction
            mults["up"] *= 1.10
            mults["down"] *= 0.90
        elif level_reaction_score < -0.6:
            # Strong bearish reaction
            mults["down"] *= 1.10
            mults["up"] *= 0.90
        
        return mults
    
    def _compute_volatility_penalty(self, volatility: float, vol_percentile: float) -> float:
        """
        Compute volatility-based uncertainty penalty.
        
        Higher volatility = more uncertainty = greater penalty.
        Extreme volatility (top percentiles) = larger penalty.
        """
        # Base penalty from current volatility
        base_penalty = volatility * 0.30  # Max 30% penalty at vol=1.0
        
        # Additional penalty for extreme volatility
        if vol_percentile > 0.85:
            extreme_boost = (vol_percentile - 0.85) * 2.0  # Extra penalty for extreme vol
            base_penalty += extreme_boost
        
        return min(base_penalty, 0.45)  # Cap at 45%
    
    def _compute_liquidity_penalty(self, liquidity_score: float) -> float:
        """
        Compute liquidity-based execution penalty.
        
        Low liquidity = higher execution costs = greater penalty.
        """
        # liquidity_score: 1 = very liquid, 0 = illiquid
        return (1.0 - liquidity_score) * 0.20  # Max 20% penalty for illiquid
    
    def _compute_regime_confidence_adjustment(self, regime_confidence: float) -> float:
        """
        Compute confidence-based adjustment to risk penalty.
        
        High confidence in regime = reduce penalty.
        Low confidence = increase penalty.
        """
        # confidence: 1 = very sure, 0 = not sure
        # adjustment: up to 15% reduction for high confidence
        return (regime_confidence ** 2) * 0.15
    
    def _compute_base_return_up(
        self,
        volatility: float,
        eval_score: float,
        probability_up: float
    ) -> float:
        """
        Compute base expected return for UP move.
        
        - Proportional to volatility (more volatile = bigger potential moves)
        - Boosted by positive eval scores
        - Scaled by scenario probability
        """
        vol_component = volatility * 0.15  # Up to 15% from volatility
        eval_component = max(0.0, eval_score * 0.10)  # Up to 10% from positive eval
        prob_component = probability_up * 0.05  # Up to 5% from probability
        
        base_return = vol_component + eval_component + prob_component
        return base_return
    
    def _compute_base_return_down(
        self,
        volatility: float,
        eval_score: float,
        probability_down: float
    ) -> float:
        """Compute base expected return for DOWN move."""
        vol_component = volatility * 0.15
        eval_component = max(0.0, -eval_score * 0.10)  # Positive for negative eval
        prob_component = probability_down * 0.05
        
        base_return = vol_component + eval_component + prob_component
        return base_return
    
    def _compute_base_return_chop(
        self,
        volatility: float,
        eval_score: float,
        probability_chop: float
    ) -> float:
        """
        Compute base expected return for CHOP (mean-reversion).
        
        Chop has smaller expected returns but more stable.
        """
        vol_component = volatility * 0.08  # Half the directional return
        # Small mean-reversion premium: neutral eval scores better in chop
        eval_component = (1.0 - abs(eval_score)) * 0.03  # Up to 3% for neutral
        prob_component = probability_chop * 0.03
        
        base_return = vol_component + eval_component + prob_component
        return base_return
    
    def _build_move_expectation(
        self,
        move_type: MoveType,
        base_return: float,
        regime_mult: float,
        flow_mult: float,
        vol_penalty: float,
        liq_penalty: float,
        total_risk_penalty: float,
        probability: float
    ) -> MoveExpectation:
        """Build complete MoveExpectation with all components."""
        # Apply regime and flow multipliers
        risk_adjusted_return = base_return * regime_mult * flow_mult
        
        # Apply penalties (fractional reduction)
        risk_adjusted_return -= vol_penalty
        risk_adjusted_return -= liq_penalty
        
        # Ensure non-negative
        risk_adjusted_return = max(0.0, risk_adjusted_return)
        
        # Compute expected value (risk-adjusted return * probability)
        expected_value = risk_adjusted_return * probability
        
        return MoveExpectation(
            move_type=move_type,
            base_return=base_return,
            regime_multiplier=regime_mult,
            flow_multiplier=flow_mult,
            volatility_penalty=vol_penalty,
            liquidity_penalty=liq_penalty,
            risk_penalty=total_risk_penalty,
            risk_adjusted_return=risk_adjusted_return,
            probability=probability,
            expected_value=expected_value
        )
    
    def _find_best_move(
        self,
        up_exp: MoveExpectation,
        down_exp: MoveExpectation,
        chop_exp: MoveExpectation
    ) -> Tuple[MoveExpectation, MoveType, float]:
        """Find move with highest expected value."""
        moves = [
            (up_exp, MoveType.UP, up_exp.expected_value),
            (down_exp, MoveType.DOWN, down_exp.expected_value),
            (chop_exp, MoveType.CHOP, chop_exp.expected_value),
        ]
        
        best_exp, best_type, best_ev = max(moves, key=lambda x: x[2])
        return best_exp, best_type, best_ev
    
    def _compute_distribution_confidence(
        self,
        regime_confidence: float,
        max_scenario_prob: float,
        combined_ev_magnitude: float
    ) -> float:
        """
        Compute overall confidence in the distribution.
        
        Higher when:
        - Regime confidence is high
        - Scenario probabilities are concentrated (not uniform)
        - Combined EV is large (not ambiguous)
        """
        regime_component = regime_confidence * 0.4
        
        # Scenario concentration (0 if uniform, 1 if concentrated)
        scenario_component = max_scenario_prob * 0.3
        
        # EV magnitude component (0 if near-zero, 1 if strong)
        # Assume max EV magnitude is ~0.15
        ev_component = min(combined_ev_magnitude / 0.15, 1.0) * 0.3
        
        confidence = regime_component + scenario_component + ev_component
        return min(confidence, 1.0)
    
    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[PROBMODEL] {message}")
