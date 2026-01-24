"""
Regime-Conditioned Scenario Simulator (Phase v2.2).

Generates price scenarios (up-move, down-move, chop) that reflect the current
market regime (TREND, RANGE, REVERSAL), enabling regime-aware expected value
calculations in the evaluator.

Philosophy:
- In TREND regimes: Continuation paths dominate, reversals are unlikely
- In RANGE regimes: Oscillation around support/resistance, symmetric probabilities
- In REVERSAL regimes: Exhaustion → reversion paths dominate, continuation is risky

This enables the evaluator's "search tree" to be regime-aware.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
from datetime import datetime


class ScenarioType(Enum):
    """Types of price scenarios."""
    UP_MOVE = "up_move"           # Strong bullish move
    DOWN_MOVE = "down_move"       # Strong bearish move
    CHOP = "chop"                 # Oscillation/ranging


@dataclass
class PriceScenario:
    """Single price movement scenario."""
    scenario_type: ScenarioType
    target_price: float
    probability: float            # Regime-adjusted probability [0, 1]
    max_drawdown: float           # Max adverse move from entry
    expected_move: float          # Expected move magnitude
    volatility_imprint: float     # Volatility impact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenario_type': self.scenario_type.value,
            'target_price': self.target_price,
            'probability': self.probability,
            'max_drawdown': self.max_drawdown,
            'expected_move': self.expected_move,
            'volatility_imprint': self.volatility_imprint,
        }


@dataclass
class ScenarioResult:
    """Result of scenario simulation with regime context."""
    scenarios: List[PriceScenario]
    expected_price: float         # Weighted average target price
    probability_up: float         # Prob(up move)
    probability_down: float       # Prob(down move)
    probability_chop: float       # Prob(chop/range)
    
    # NEW: Regime-specific fields (Phase v2.2)
    regime_label: str = ""
    regime_confidence: float = 0.0
    scenario_bias: str = ""       # "bullish", "bearish", "symmetric"
    regime_alignment: float = 0.0 # 0-1: how well scenarios match regime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenarios': [s.to_dict() for s in self.scenarios],
            'expected_price': self.expected_price,
            'probability_up': self.probability_up,
            'probability_down': self.probability_down,
            'probability_chop': self.probability_chop,
            'regime_label': self.regime_label,
            'regime_confidence': self.regime_confidence,
            'scenario_bias': self.scenario_bias,
            'regime_alignment': self.regime_alignment,
        }


class ScenarioSimulator:
    """
    Generates regime-conditioned price scenarios.
    
    Uses regime classification to bias scenario probabilities:
    - TREND: Favors continuation (up in bull, down in bear)
    - RANGE: Symmetric probabilities, favors chop
    - REVERSAL: Favors reversal scenarios
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize scenario simulator."""
        self.verbose = verbose
    
    def simulate_scenarios(
        self,
        current_price: float,
        vwap: float,
        session_high: float,
        session_low: float,
        expected_move: float,
        volatility: float,
        regime_label: str = "",
        regime_confidence: float = 0.0,
        eval_score: float = 0.0
    ) -> ScenarioResult:
        """
        Simulate price scenarios conditioned on regime.
        
        Args:
            current_price: Current market price
            vwap: Volume-weighted average price (key support/resistance in RANGE/REVERSAL)
            session_high/low: Session extremes
            expected_move: Base expected move magnitude
            volatility: Current volatility estimate
            regime_label: TREND, RANGE, REVERSAL, or UNKNOWN
            regime_confidence: 0-1, strength of regime classification
            eval_score: Evaluation score [-1, +1] indicating direction bias
        
        Returns:
            ScenarioResult with regime-weighted scenarios
        """
        
        # Base scenario parameters
        up_move_magnitude = expected_move * 1.2
        down_move_magnitude = expected_move * 1.2
        chop_move_magnitude = expected_move * 0.6
        
        # Generate base scenarios
        up_scenario = self._create_up_scenario(
            current_price, vwap, up_move_magnitude, volatility
        )
        down_scenario = self._create_down_scenario(
            current_price, vwap, down_move_magnitude, volatility
        )
        chop_scenario = self._create_chop_scenario(
            current_price, vwap, chop_move_magnitude, volatility
        )
        
        # Apply regime conditioning
        if regime_label == "TREND":
            probs_up, probs_down, probs_chop, bias = self._condition_for_trend(
                eval_score, regime_confidence
            )
        elif regime_label == "RANGE":
            probs_up, probs_down, probs_chop, bias = self._condition_for_range(
                regime_confidence
            )
        elif regime_label == "REVERSAL":
            probs_up, probs_down, probs_chop, bias = self._condition_for_reversal(
                eval_score, regime_confidence
            )
        else:
            # Default: neutral probabilities
            probs_up, probs_down, probs_chop, bias = 0.33, 0.33, 0.34, "symmetric"
        
        # Create scenarios with regime-adjusted probabilities
        up_scenario.probability = probs_up
        down_scenario.probability = probs_down
        chop_scenario.probability = probs_chop
        
        scenarios = [up_scenario, down_scenario, chop_scenario]
        
        # Calculate expected price
        expected_price = (
            up_scenario.target_price * probs_up +
            down_scenario.target_price * probs_down +
            chop_scenario.target_price * probs_chop
        )
        
        # Calculate regime alignment (0-1): how well scenarios match regime
        regime_alignment = self._calculate_alignment(
            regime_label, eval_score, regime_confidence
        )
        
        result = ScenarioResult(
            scenarios=scenarios,
            expected_price=expected_price,
            probability_up=probs_up,
            probability_down=probs_down,
            probability_chop=probs_chop,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            scenario_bias=bias,
            regime_alignment=regime_alignment,
        )
        
        return result
    
    def _create_up_scenario(
        self,
        current_price: float,
        vwap: float,
        move_magnitude: float,
        volatility: float
    ) -> PriceScenario:
        """Create up-move scenario."""
        target = current_price + move_magnitude
        max_dd = -move_magnitude * 0.3  # Up scenarios: light pullback
        
        return PriceScenario(
            scenario_type=ScenarioType.UP_MOVE,
            target_price=target,
            probability=0.33,  # Will be overridden by regime conditioning
            max_drawdown=max_dd,
            expected_move=move_magnitude,
            volatility_imprint=volatility * 1.1,
        )
    
    def _create_down_scenario(
        self,
        current_price: float,
        vwap: float,
        move_magnitude: float,
        volatility: float
    ) -> PriceScenario:
        """Create down-move scenario."""
        target = current_price - move_magnitude
        max_dd = move_magnitude * 0.3  # Down scenarios: light bounce
        
        return PriceScenario(
            scenario_type=ScenarioType.DOWN_MOVE,
            target_price=target,
            probability=0.33,
            max_drawdown=max_dd,
            expected_move=move_magnitude,
            volatility_imprint=volatility * 1.1,
        )
    
    def _create_chop_scenario(
        self,
        current_price: float,
        vwap: float,
        move_magnitude: float,
        volatility: float
    ) -> PriceScenario:
        """Create oscillation/ranging scenario."""
        # Chop orbits around VWAP
        target = vwap
        max_dd = move_magnitude
        
        return PriceScenario(
            scenario_type=ScenarioType.CHOP,
            target_price=target,
            probability=0.34,
            max_drawdown=-max_dd,  # Two-sided
            expected_move=move_magnitude * 0.5,
            volatility_imprint=volatility * 0.7,
        )
    
    def _condition_for_trend(
        self,
        eval_score: float,
        regime_confidence: float
    ) -> tuple:
        """
        TREND regime: Continuation paths dominate.
        - Bullish eval (>0): Up moves weighted heavily, down moves low
        - Bearish eval (<0): Down moves weighted heavily, up moves low
        - Confidence: Higher confidence → more skewed probabilities
        
        Returns: (prob_up, prob_down, prob_chop, bias_label)
        """
        
        # Base: favor continuation by 60% total
        if eval_score > 0:
            # Bullish: strong up bias
            base_up = 0.55
            base_down = 0.15
            base_chop = 0.30
            bias = "bullish"
        else:
            # Bearish: strong down bias
            base_up = 0.15
            base_down = 0.55
            base_chop = 0.30
            bias = "bearish"
        
        # Apply confidence weighting (higher confidence → stronger skew)
        conf_factor = regime_confidence if regime_confidence > 0.3 else 1.0
        skew_amount = abs(eval_score) * 0.15 * conf_factor
        
        if eval_score > 0:
            prob_up = min(base_up + skew_amount, 0.75)
            prob_down = max(base_down - skew_amount * 0.5, 0.05)
            prob_chop = 1.0 - prob_up - prob_down
        else:
            prob_down = min(base_down + skew_amount, 0.75)
            prob_up = max(base_up - skew_amount * 0.5, 0.05)
            prob_chop = 1.0 - prob_up - prob_down
        
        # Normalize
        total = prob_up + prob_down + prob_chop
        prob_up /= total
        prob_down /= total
        prob_chop /= total
        
        return prob_up, prob_down, prob_chop, bias
    
    def _condition_for_range(
        self,
        regime_confidence: float
    ) -> tuple:
        """
        RANGE regime: Oscillation around VWAP, symmetric up/down.
        - Chop probability elevated (fails breakouts)
        - Up and down equally likely
        - Confidence: Higher confidence → more chop, less directional
        
        Returns: (prob_up, prob_down, prob_chop, bias_label)
        """
        
        # Base: symmetric with bias to chop
        base_up = 0.25
        base_down = 0.25
        base_chop = 0.50
        
        # Confidence weighting: high confidence in RANGE → even more chop
        if regime_confidence > 0.5:
            chop_boost = regime_confidence * 0.15
            prob_chop = min(base_chop + chop_boost, 0.70)
            prob_up = (1.0 - prob_chop) * 0.5
            prob_down = (1.0 - prob_chop) * 0.5
        else:
            prob_up = base_up
            prob_down = base_down
            prob_chop = base_chop
        
        # Normalize
        total = prob_up + prob_down + prob_chop
        prob_up /= total
        prob_down /= total
        prob_chop /= total
        
        return prob_up, prob_down, prob_chop, "symmetric"
    
    def _condition_for_reversal(
        self,
        eval_score: float,
        regime_confidence: float
    ) -> tuple:
        """
        REVERSAL regime: Exhaustion → reversion paths dominate.
        - High eval_score (strong up): Reversal down is likely
        - Low eval_score (strong down): Reversal up is likely
        - Continuation probability reduced
        - Confidence: Higher confidence → stronger reversal bias
        
        Returns: (prob_up, prob_down, prob_chop, bias_label)
        """
        
        # Base: reversal bias (opposite of eval_score direction)
        if eval_score > 0.3:
            # Strong up exhaustion → down reversal
            base_up = 0.10
            base_down = 0.60
            base_chop = 0.30
            bias = "bearish_reversal"
        elif eval_score < -0.3:
            # Strong down exhaustion → up reversal
            base_up = 0.60
            base_down = 0.10
            base_chop = 0.30
            bias = "bullish_reversal"
        else:
            # Neutral: symmetric reversal
            base_up = 0.35
            base_down = 0.35
            base_chop = 0.30
            bias = "symmetric_reversal"
        
        # Apply confidence weighting
        conf_factor = regime_confidence if regime_confidence > 0.3 else 1.0
        reversal_boost = 0.15 * conf_factor
        
        if eval_score > 0.3:
            prob_down = min(base_down + reversal_boost, 0.75)
            prob_up = max(base_up - reversal_boost * 0.5, 0.05)
            prob_chop = 1.0 - prob_up - prob_down
        elif eval_score < -0.3:
            prob_up = min(base_up + reversal_boost, 0.75)
            prob_down = max(base_down - reversal_boost * 0.5, 0.05)
            prob_chop = 1.0 - prob_up - prob_down
        else:
            prob_up = base_up
            prob_down = base_down
            prob_chop = base_chop
        
        # Normalize
        total = prob_up + prob_down + prob_chop
        prob_up /= total
        prob_down /= total
        prob_chop /= total
        
        return prob_up, prob_down, prob_chop, bias
    
    def _calculate_alignment(
        self,
        regime_label: str,
        eval_score: float,
        regime_confidence: float
    ) -> float:
        """
        Calculate scenario alignment with regime (0-1).
        
        Higher alignment means scenarios strongly reflect the regime.
        Used by evaluator to adjust confidence based on alignment strength.
        """
        
        if regime_confidence < 0.3:
            return 0.3  # Low confidence regime → weak alignment
        
        if regime_label == "TREND":
            # Alignment: eval_score magnitude indicates alignment
            return min(abs(eval_score) * regime_confidence, 1.0)
        elif regime_label == "RANGE":
            # Alignment: always high in RANGE (symmetric is predictable)
            return 0.7 + regime_confidence * 0.2
        elif regime_label == "REVERSAL":
            # Alignment: extreme eval_score indicates exhaustion
            return min(abs(eval_score) * regime_confidence, 1.0)
        else:
            # UNKNOWN regime
            return 0.5
