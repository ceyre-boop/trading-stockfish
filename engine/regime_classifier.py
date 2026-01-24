"""
Regime Classifier (Phase v1.2).

Deterministic, causal classification of trading days as:
  - TREND: Persistent directional movement
  - REVERSAL: Failed breakouts, directional exhaustion
  - RANGE: Mean-reversion, oscillation, compression

Uses only causal signals (no future data lookahead).
"""

from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
from datetime import datetime
import logging


@dataclass
class RegimeSignal:
    """Individual regime indicator."""
    name: str
    value: float  # 0.0–1.0, contribution to regime
    regime_type: str  # 'TREND', 'REVERSAL', 'RANGE'
    reasoning: str = ""


@dataclass
class RegimeState:
    """Current regime classification."""
    regime_label: str  # 'TREND', 'REVERSAL', 'RANGE'
    regime_confidence: float  # 0.0–1.0
    regime_features: Dict[str, Any] = field(default_factory=dict)
    contributing_signals: List[RegimeSignal] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime_label': self.regime_label,
            'regime_confidence': self.regime_confidence,
            'regime_features': self.regime_features,
            'contributing_signals': [
                {
                    'name': s.name,
                    'value': s.value,
                    'regime_type': s.regime_type,
                    'reasoning': s.reasoning,
                }
                for s in self.contributing_signals
            ],
            'timestamp': self.timestamp.isoformat(),
        }


class RegimeClassifier:
    """
    Classify trading days as TREND, REVERSAL, or RANGE.
    
    Rules:
    - TREND: Persistent directional movement with higher highs/lows or strong initiative
    - REVERSAL: Failed breakouts, exhaustion, VWAP reversion after extension
    - RANGE: Oscillation around VWAP, low initiative, compression windows
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize regime classifier."""
        self.logger = logger
        
        # Running state for regime inference
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.vwap_history: List[float] = []
        self.initiative_history: List[bool] = []
        self.high_lows_history: List[Tuple[float, float]] = []  # Track HH/LL
        
        # Session state
        self.session_high: float = 0.0
        self.session_low: float = float('inf')
        self.overnight_high: float = 0.0
        self.overnight_low: float = float('inf')
        
        self.current_regime = RegimeState(
            regime_label='RANGE',
            regime_confidence=0.5
        )
    
    def update_with_bar(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        vwap: float,
        flow_signals: Optional[Dict[str, Any]] = None,
        session: str = 'RTH',
        initiative_detected: bool = False,
        stop_run_detected: bool = False
    ) -> RegimeState:
        """
        Update regime classification with new bar.
        
        Args:
            timestamp: Bar timestamp
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            vwap: VWAP for bar
            flow_signals: Optional flow context (initiative, stop-run, etc.)
            session: Trading session (e.g., 'RTH', 'Globex')
            initiative_detected: Whether initiative move detected
            stop_run_detected: Whether stop-run detected
        
        Returns:
            Current regime state
        """
        # Update history (keep last 120 bars for analysis)
        self.price_history.append(close)
        self.high_history.append(high)
        self.low_history.append(low)
        self.volume_history.append(volume)
        self.vwap_history.append(vwap)
        self.high_lows_history.append((high, low))
        
        if len(self.price_history) > 120:
            self.price_history.pop(0)
            self.high_history.pop(0)
            self.low_history.pop(0)
            self.volume_history.pop(0)
            self.vwap_history.pop(0)
            self.high_lows_history.pop(0)
        
        # Update session extremes
        self.session_high = max(self.session_high, high)
        self.session_low = min(self.session_low, low)
        
        # Extract flow context
        flow_signals = flow_signals or {}
        initiative_signals = initiative_detected or flow_signals.get('initiative_move_detected', False)
        stop_run_signals = stop_run_detected or flow_signals.get('stop_run_detected', False)
        
        self.initiative_history.append(initiative_signals)
        if len(self.initiative_history) > 120:
            self.initiative_history.pop(0)
        
        # Classify regime
        self.current_regime = self._classify_regime(
            close, high, low, vwap, initiative_signals, stop_run_signals
        )
        
        return self.current_regime
    
    def _classify_regime(
        self,
        close: float,
        high: float,
        low: float,
        vwap: float,
        initiative: bool,
        stop_run: bool
    ) -> RegimeState:
        """
        Classify regime based on signals.
        
        Returns:
            RegimeState with classification and confidence
        """
        signals = []
        
        # Need at least 20 bars for reliable classification
        if len(self.price_history) < 20:
            return RegimeState(
                regime_label='RANGE',
                regime_confidence=0.3,
                regime_features={'reason': 'insufficient_bars'},
                contributing_signals=signals
            )
        
        # Calculate regime indicators
        
        # 1. VWAP Distance & Persistence (Trend indicator)
        vwap_distance = self._calculate_vwap_distance(close, vwap)
        vwap_persistence = self._calculate_vwap_persistence()
        
        if vwap_distance > 0.015 and vwap_persistence > 0.6:
            signals.append(RegimeSignal(
                name='VWAP_PERSISTENCE',
                value=0.7,
                regime_type='TREND',
                reasoning=f'VWAP distance {vwap_distance:.2%}, persistence {vwap_persistence:.0%}'
            ))
        
        # 2. Higher Highs / Higher Lows (Trend indicator)
        hh_hl_score = self._calculate_higher_highs_lows()
        if hh_hl_score > 0.6:
            signals.append(RegimeSignal(
                name='HIGHER_HIGHS_LOWS',
                value=0.7,
                regime_type='TREND',
                reasoning=f'HH/HL pattern score: {hh_hl_score:.0%}'
            ))
        elif hh_hl_score < 0.4:
            signals.append(RegimeSignal(
                name='LOWER_HIGHS_LOWS',
                value=0.7,
                regime_type='TREND',
                reasoning=f'LL/LH pattern score: {hh_hl_score:.0%}'
            ))
        
        # 3. Initiative Moves (Trend indicator)
        initiative_ratio = self._calculate_initiative_ratio()
        if initiative_ratio > 0.5:
            signals.append(RegimeSignal(
                name='SUSTAINED_INITIATIVE',
                value=0.6,
                regime_type='TREND',
                reasoning=f'Initiative ratio: {initiative_ratio:.0%}'
            ))
        
        # 4. Stop-Run Detection (Reversal indicator)
        if stop_run:
            signals.append(RegimeSignal(
                name='STOP_RUN_DETECTED',
                value=0.5,
                regime_type='REVERSAL',
                reasoning='Stop-run signal exhaustion'
            ))
        
        # 5. VWAP Reversion (Reversal indicator)
        vwap_reversion = self._calculate_vwap_reversion()
        if vwap_reversion > 0.6:
            signals.append(RegimeSignal(
                name='VWAP_REVERSION',
                value=0.65,
                regime_type='REVERSAL',
                reasoning=f'Strong VWAP mean-reversion: {vwap_reversion:.0%}'
            ))
        
        # 6. Failed Breakout (Reversal indicator)
        failed_breakout_score = self._calculate_failed_breakout_score()
        if failed_breakout_score > 0.6:
            signals.append(RegimeSignal(
                name='FAILED_BREAKOUT',
                value=0.65,
                regime_type='REVERSAL',
                reasoning=f'Breakout rejection score: {failed_breakout_score:.0%}'
            ))
        
        # 7. Volatility Compression (Range indicator)
        compression_score = self._calculate_compression()
        if compression_score > 0.6:
            signals.append(RegimeSignal(
                name='COMPRESSION_WINDOW',
                value=0.6,
                regime_type='RANGE',
                reasoning=f'Volatility compression: {compression_score:.0%}'
            ))
        
        # 8. Low Initiative (Range indicator)
        if initiative_ratio < 0.3:
            signals.append(RegimeSignal(
                name='LOW_INITIATIVE',
                value=0.55,
                regime_type='RANGE',
                reasoning=f'Low initiative ratio: {initiative_ratio:.0%}'
            ))
        
        # 9. Oscillation Around VWAP (Range indicator)
        oscillation_score = self._calculate_oscillation_score()
        if oscillation_score > 0.65:
            signals.append(RegimeSignal(
                name='OSCILLATION_AROUND_VWAP',
                value=0.65,
                regime_type='RANGE',
                reasoning=f'Oscillation score: {oscillation_score:.0%}'
            ))
        
        # Aggregate regime classification
        regime_label, regime_confidence = self._aggregate_signals(signals)
        
        return RegimeState(
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            regime_features={
                'vwap_distance': vwap_distance,
                'vwap_persistence': vwap_persistence,
                'hh_hl_score': hh_hl_score,
                'initiative_ratio': initiative_ratio,
                'vwap_reversion': vwap_reversion,
                'failed_breakout_score': failed_breakout_score,
                'compression_score': compression_score,
                'oscillation_score': oscillation_score,
            },
            contributing_signals=signals
        )
    
    def _calculate_vwap_distance(self, close: float, vwap: float) -> float:
        """Calculate distance from VWAP as percentage."""
        if vwap == 0:
            return 0.0
        return abs(close - vwap) / vwap
    
    def _calculate_vwap_persistence(self) -> float:
        """
        Calculate how long price stays away from VWAP.
        Returns 0-1 (0 = always at VWAP, 1 = always away from VWAP).
        """
        if len(self.price_history) < 10:
            return 0.5
        
        # Check last 10 bars
        away_from_vwap = 0
        for i in range(-10, 0):
            dist = abs(self.price_history[i] - self.vwap_history[i]) / self.vwap_history[i]
            if dist > 0.01:
                away_from_vwap += 1
        
        return away_from_vwap / 10.0
    
    def _calculate_higher_highs_lows(self) -> float:
        """
        Calculate higher highs & higher lows score.
        Returns 0-1 (0 = lower highs/lows, 1 = higher highs/lows).
        """
        if len(self.high_history) < 20:
            return 0.5
        
        # Check last 20 bars for trend
        higher_highs = 0
        higher_lows = 0
        
        for i in range(-20, -1):
            prev_idx = i - 1
            # Ensure we don't go out of bounds
            if prev_idx >= -len(self.high_history):
                if self.high_history[i] > self.high_history[prev_idx]:
                    higher_highs += 1
                if self.low_history[i] > self.low_history[prev_idx]:
                    higher_lows += 1
        
        # Score: (HH + HL) / (total bars)
        trend_score = (higher_highs + higher_lows) / 38.0
        
        # Normalize: 0.5 = neutral, 1.0 = all higher, 0.0 = all lower
        return trend_score
    
    def _calculate_initiative_ratio(self) -> float:
        """
        Calculate ratio of bars with initiative signals.
        Returns 0-1 (0 = no initiative, 1 = all bars have initiative).
        """
        if len(self.initiative_history) < 10:
            return 0.0
        
        # Check last 20 bars
        recent = self.initiative_history[-20:] if len(self.initiative_history) >= 20 else self.initiative_history
        return sum(recent) / len(recent)
    
    def _calculate_vwap_reversion(self) -> float:
        """
        Calculate strength of mean-reversion to VWAP.
        Returns 0-1 (high score = strong reversion tendency).
        """
        if len(self.price_history) < 20:
            return 0.5
        
        # Count reversions: price extends away then comes back to VWAP
        reversions = 0
        for i in range(-10, 0):
            dist_now = abs(self.price_history[i] - self.vwap_history[i])
            dist_prev = abs(self.price_history[i-1] - self.vwap_history[i-1])
            
            # Reverting if distance decreased
            if dist_now < dist_prev:
                reversions += 1
        
        return reversions / 10.0
    
    def _calculate_failed_breakout_score(self) -> float:
        """
        Calculate score for failed breakouts.
        Returns 0-1 (high score = many failed breakouts).
        """
        if len(self.high_history) < 20:
            return 0.5
        
        # Detect failed breakouts: high > session_high_20bars, then reverses
        failed_breakouts = 0
        recent_20_high = max(self.high_history[-20:-1]) if len(self.high_history) >= 20 else max(self.high_history)
        
        for i in range(-10, -1):
            if i < -1:
                # Ensure we have valid indices
                if abs(i + 1) <= len(self.price_history):
                    if self.high_history[i] > recent_20_high and self.price_history[i] > self.price_history[i+1]:
                        failed_breakouts += 1
        
        return min(failed_breakouts / 5.0, 1.0)
    
    @property
    def close_history(self) -> List[float]:
        """Get close history."""
        return self.price_history
    
    def _calculate_compression(self) -> float:
        """
        Calculate volatility compression (low ATR relative to history).
        Returns 0-1 (high = compressed volatility).
        """
        if len(self.high_history) < 20:
            return 0.5
        
        # Calculate recent ATR
        recent_atr = self._calculate_atr(10)
        historical_atr = self._calculate_atr(20)
        
        if historical_atr == 0:
            return 0.5
        
        atr_ratio = recent_atr / historical_atr
        
        # Compression if recent ATR < 0.7 * historical ATR
        compression = max(0, 1 - atr_ratio)
        return min(compression, 1.0)
    
    def _calculate_atr(self, periods: int) -> float:
        """Calculate Average True Range."""
        if len(self.high_history) < periods:
            return 0.0
        
        trs = []
        for i in range(-periods, 0):
            high = self.high_history[i]
            low = self.low_history[i]
            prev_close = self.price_history[i-1] if i > -len(self.price_history) else self.price_history[i]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)
        
        return np.mean(trs) if trs else 0.0
    
    def _calculate_oscillation_score(self) -> float:
        """
        Calculate oscillation around VWAP (mean-reversion tendency).
        Returns 0-1 (high = strong oscillation).
        """
        if len(self.price_history) < 20:
            return 0.5
        
        # Count crossings of VWAP in last 20 bars
        crossings = 0
        for i in range(-20, -1):
            # Check bounds
            if len(self.price_history) < abs(i) + 1:
                continue
            
            above_now = self.price_history[i] > self.vwap_history[i]
            above_prev = self.price_history[i-1] > self.vwap_history[i-1]
            
            if above_now != above_prev:
                crossings += 1
        
        # More crossings = more oscillation
        return min(crossings / 20.0, 1.0)
    
    def _aggregate_signals(
        self,
        signals: List[RegimeSignal]
    ) -> Tuple[str, float]:
        """
        Aggregate signals into regime classification.
        
        Returns:
            (regime_label, confidence)
        """
        if not signals:
            return 'RANGE', 0.5
        
        # Sum scores by regime type
        trend_score = sum(s.value for s in signals if s.regime_type == 'TREND')
        reversal_score = sum(s.value for s in signals if s.regime_type == 'REVERSAL')
        range_score = sum(s.value for s in signals if s.regime_type == 'RANGE')
        
        # Normalize scores
        total = trend_score + reversal_score + range_score
        if total == 0:
            return 'RANGE', 0.5
        
        trend_pct = trend_score / total
        reversal_pct = reversal_score / total
        range_pct = range_score / total
        
        # Choose regime with highest score
        max_score = max(trend_score, reversal_score, range_score)
        
        if max_score == trend_score:
            regime = 'TREND'
            confidence = trend_pct
        elif max_score == reversal_score:
            regime = 'REVERSAL'
            confidence = reversal_pct
        else:
            regime = 'RANGE'
            confidence = range_pct
        
        return regime, confidence
    
    def get_regime_state(self) -> RegimeState:
        """Get current regime state."""
        return self.current_regime
    
    def reset_session(self):
        """Reset for new trading session."""
        self.session_high = 0.0
        self.session_low = float('inf')
        self.overnight_high = self.session_high
        self.overnight_low = self.session_low
        
        # Keep price history but not session state
        if self.logger:
            self.logger.info(
                f"Session reset. Last regime: {self.current_regime.regime_label} "
                f"({self.current_regime.regime_confidence:.0%})"
            )
    
    def reset(self):
        """Reset classifier for new day/analysis."""
        self.price_history.clear()
        self.volume_history.clear()
        self.vwap_history.clear()
        self.high_lows_history.clear()
        self.initiative_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.session_high = 0.0
        self.session_low = float('inf')
        self.overnight_high = 0.0
        self.overnight_low = float('inf')
        self.current_regime = RegimeState(
            regime_label='RANGE',
            regime_confidence=0.3,
            regime_features={},
            contributing_signals=[]
        )
