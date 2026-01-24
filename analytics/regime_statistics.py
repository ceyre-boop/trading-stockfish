"""
Regime Statistics Module (Phase v2.0).

Statistical validation and analysis of regime classifications across
multiple historical days. Computes distributions, transitions, and
feature summaries without lookahead bias.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from engine.regime_classifier import RegimeClassifier, RegimeState


@dataclass
class RegimeTransition:
    """Record of a regime transition."""
    from_regime: Optional[str]
    to_regime: str
    timestamp: datetime
    price: float
    confidence: float
    bar_index: int


@dataclass
class DayRegimeStats:
    """Statistics for a single day's regime classification."""
    date: datetime
    regime_counts: Dict[str, int] = field(default_factory=dict)  # minutes per regime
    regime_confidence_avg: Dict[str, float] = field(default_factory=dict)  # avg confidence
    transitions: List[RegimeTransition] = field(default_factory=list)
    dominant_regime: Optional[str] = None
    dominant_regime_pct: float = 0.0
    transition_count: int = 0
    feature_averages: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'date': self.date.isoformat(),
            'regime_counts': self.regime_counts,
            'regime_confidence_avg': {
                k: f"{v:.1%}" for k, v in self.regime_confidence_avg.items()
            },
            'dominant_regime': self.dominant_regime,
            'dominant_regime_pct': f"{self.dominant_regime_pct:.1%}",
            'transition_count': self.transition_count,
            'feature_averages': {
                k: f"{v:.3f}" for k, v in self.feature_averages.items()
            },
        }


@dataclass
class AggregateRegimeStats:
    """Aggregated statistics across multiple days."""
    total_days: int
    total_bars: int
    regime_frequency: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    transition_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    confidence_distribution: Dict[str, List[float]] = field(default_factory=dict)
    feature_distributions: Dict[str, List[float]] = field(default_factory=dict)
    average_duration: Dict[str, float] = field(default_factory=dict)
    day_type_distribution: Dict[str, int] = field(default_factory=dict)


class RegimeStatistics:
    """Statistical analysis of regime classifications across multiple days."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize regime statistics analyzer."""
        self.logger = logger or self._setup_logger()
        self.classifier = RegimeClassifier(logger=self.logger)
        self.day_stats: List[DayRegimeStats] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger('RegimeStats')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger
    
    def analyze_day(
        self,
        day_data: pd.DataFrame,
        day_date: datetime,
        symbol: str = 'ES'
    ) -> DayRegimeStats:
        """
        Analyze regime classification for a single day.
        
        Args:
            day_data: DataFrame with OHLCV and VWAP columns
            day_date: Date of the day being analyzed
            symbol: Symbol being analyzed (ES/NQ)
            
        Returns:
            DayRegimeStats with regime analysis for the day
        """
        self.classifier.reset()
        
        stats = DayRegimeStats(date=day_date)
        regime_counts = defaultdict(int)
        regime_confidences = defaultdict(list)
        regime_features = defaultdict(list)
        
        previous_regime = None
        bars_in_regime = defaultdict(int)
        regime_start_bar = 0
        
        for idx, row in day_data.iterrows():
            # Update classifier
            regime_state = self.classifier.update_with_bar(
                open_price=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                vwap=row['vwap'],
                timestamp=row.get('timestamp', day_date + timedelta(minutes=idx)),
                session=row.get('session', 'RTH'),
                initiative_detected=row.get('initiative', False),
                stop_run_detected=row.get('stop_run', False),
                flow_signals=row.get('flow_signals', {})
            )
            
            # Track regime counts
            regime_counts[regime_state.regime_label] += 1
            regime_confidences[regime_state.regime_label].append(
                regime_state.regime_confidence
            )
            
            # Track features
            for feature_name, feature_value in regime_state.regime_features.items():
                if isinstance(feature_value, (int, float)):
                    regime_features[feature_name].append(feature_value)
            
            # Detect regime transitions
            if previous_regime != regime_state.regime_label:
                if previous_regime is not None:
                    # Record transition
                    transition = RegimeTransition(
                        from_regime=previous_regime,
                        to_regime=regime_state.regime_label,
                        timestamp=row.get('timestamp', day_date + timedelta(minutes=idx)),
                        price=row['close'],
                        confidence=regime_state.regime_confidence,
                        bar_index=idx
                    )
                    stats.transitions.append(transition)
                    stats.transition_count += 1
                
                previous_regime = regime_state.regime_label
                regime_start_bar = idx
            
            bars_in_regime[regime_state.regime_label] += 1
        
        # Populate stats
        stats.regime_counts = dict(regime_counts)
        
        # Calculate average confidence per regime
        for regime, confidences in regime_confidences.items():
            stats.regime_confidence_avg[regime] = np.mean(confidences)
        
        # Calculate feature averages
        for feature_name, values in regime_features.items():
            if values:
                stats.feature_averages[feature_name] = np.mean(values)
        
        # Determine dominant regime
        if regime_counts:
            dominant = max(regime_counts.items(), key=lambda x: x[1])
            stats.dominant_regime = dominant[0]
            stats.dominant_regime_pct = dominant[1] / len(day_data)
        
        self.day_stats.append(stats)
        return stats
    
    def aggregate_statistics(self) -> AggregateRegimeStats:
        """
        Aggregate statistics across all analyzed days.
        
        Returns:
            AggregateRegimeStats with cross-day aggregations
        """
        if not self.day_stats:
            self.logger.warning("No day statistics to aggregate")
            return AggregateRegimeStats(total_days=0, total_bars=0)
        
        aggregate = AggregateRegimeStats(
            total_days=len(self.day_stats),
            total_bars=sum(sum(d.regime_counts.values()) for d in self.day_stats)
        )
        
        # Regime frequency
        regime_bar_counts = defaultdict(int)
        regime_day_counts = defaultdict(int)
        regime_confidence_lists = defaultdict(list)
        regime_feature_lists = defaultdict(lambda: defaultdict(list))
        
        for day_stat in self.day_stats:
            # Track which days had each regime
            for regime in day_stat.regime_counts.keys():
                regime_day_counts[regime] += 1
            
            # Track bar counts and confidences
            for regime, count in day_stat.regime_counts.items():
                regime_bar_counts[regime] += count
            
            for regime, avg_conf in day_stat.regime_confidence_avg.items():
                regime_confidence_lists[regime].append(avg_conf)
            
            for feature_name, value in day_stat.feature_averages.items():
                regime_feature_lists['all_regimes'][feature_name].append(value)
        
        # Build frequency dict
        for regime in regime_bar_counts.keys():
            aggregate.regime_frequency[regime] = {
                'bars': regime_bar_counts[regime],
                'pct_of_total_bars': regime_bar_counts[regime] / aggregate.total_bars,
                'days_with_regime': regime_day_counts[regime],
                'pct_of_days': regime_day_counts[regime] / len(self.day_stats),
                'avg_confidence': np.mean(regime_confidence_lists[regime]),
            }
        
        # Build transition matrix
        transition_counts = defaultdict(lambda: defaultdict(int))
        for day_stat in self.day_stats:
            for transition in day_stat.transitions:
                transition_counts[transition.from_regime][transition.to_regime] += 1
        
        aggregate.transition_matrix = {
            from_r: dict(to_counts)
            for from_r, to_counts in transition_counts.items()
        }
        
        # Confidence distributions
        for regime, confidences in regime_confidence_lists.items():
            aggregate.confidence_distribution[regime] = confidences
        
        # Feature distributions
        for feature_name, values in regime_feature_lists['all_regimes'].items():
            aggregate.feature_distributions[feature_name] = values
        
        # Average duration (bars per regime between transitions)
        for regime in regime_bar_counts.keys():
            total_bars_in_regime = regime_bar_counts[regime]
            transition_count = sum(1 for d in self.day_stats for t in d.transitions 
                                 if t.to_regime == regime)
            # Rough estimate: total bars / transitions into regime
            avg_duration = (total_bars_in_regime / max(transition_count, 1)
                          if transition_count > 0 else total_bars_in_regime)
            aggregate.average_duration[regime] = avg_duration
        
        # Day type distribution
        for day_stat in self.day_stats:
            if day_stat.dominant_regime:
                aggregate.day_type_distribution[day_stat.dominant_regime] = (
                    aggregate.day_type_distribution.get(day_stat.dominant_regime, 0) + 1
                )
        
        return aggregate
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report of all statistics."""
        if not self.day_stats:
            return "No statistics to report"
        
        aggregate = self.aggregate_statistics()
        
        lines = []
        lines.append("=" * 80)
        lines.append("REGIME STATISTICS SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Total Days Analyzed: {aggregate.total_days}")
        lines.append(f"Total Bars Analyzed: {aggregate.total_bars}")
        lines.append("")
        
        # Regime frequency section
        lines.append("REGIME FREQUENCY")
        lines.append("-" * 80)
        for regime, freq_data in sorted(aggregate.regime_frequency.items()):
            lines.append(f"\n{regime}:")
            lines.append(f"  Bars: {freq_data['bars']} ({freq_data['pct_of_total_bars']:.1%})")
            lines.append(f"  Days with regime: {freq_data['days_with_regime']} ({freq_data['pct_of_days']:.1%})")
            lines.append(f"  Avg confidence: {freq_data['avg_confidence']:.1%}")
        lines.append("")
        
        # Day type distribution
        lines.append("DAY TYPE DISTRIBUTION (by dominant regime)")
        lines.append("-" * 80)
        for regime, count in sorted(aggregate.day_type_distribution.items()):
            pct = count / aggregate.total_days * 100
            lines.append(f"{regime:12} {count:3} days ({pct:5.1f}%)")
        lines.append("")
        
        # Average duration
        lines.append("AVERAGE REGIME DURATION (bars)")
        lines.append("-" * 80)
        for regime, duration in sorted(aggregate.average_duration.items()):
            lines.append(f"{regime:12} {duration:6.1f} bars")
        lines.append("")
        
        # Transition matrix
        lines.append("TRANSITION MATRIX (count of transitions)")
        lines.append("-" * 80)
        
        all_regimes = set()
        for from_r, to_dict in aggregate.transition_matrix.items():
            all_regimes.add(from_r)
            all_regimes.update(to_dict.keys())
        
        all_regimes = sorted(all_regimes)
        
        # Header
        lines.append("From \\ To     " + "  ".join(f"{r:>10}" for r in all_regimes))
        
        # Rows
        for from_r in all_regimes:
            row_data = []
            for to_r in all_regimes:
                count = aggregate.transition_matrix.get(from_r, {}).get(to_r, 0)
                row_data.append(f"{count:>10}")
            lines.append(f"{from_r:<10} {' '.join(row_data)}")
        lines.append("")
        
        # Confidence statistics
        lines.append("CONFIDENCE DISTRIBUTION")
        lines.append("-" * 80)
        for regime in sorted(aggregate.confidence_distribution.keys()):
            confidences = aggregate.confidence_distribution[regime]
            if confidences:
                lines.append(f"\n{regime}:")
                lines.append(f"  Mean:    {np.mean(confidences):.1%}")
                lines.append(f"  Median:  {np.median(confidences):.1%}")
                lines.append(f"  Min:     {np.min(confidences):.1%}")
                lines.append(f"  Max:     {np.max(confidences):.1%}")
                lines.append(f"  Std Dev: {np.std(confidences):.1%}")
        lines.append("")
        
        # Feature statistics
        if aggregate.feature_distributions:
            lines.append("FEATURE STATISTICS")
            lines.append("-" * 80)
            for feature_name in sorted(aggregate.feature_distributions.keys()):
                values = aggregate.feature_distributions[feature_name]
                if values:
                    lines.append(f"\n{feature_name}:")
                    lines.append(f"  Mean:    {np.mean(values):.4f}")
                    lines.append(f"  Median:  {np.median(values):.4f}")
                    lines.append(f"  Min:     {np.min(values):.4f}")
                    lines.append(f"  Max:     {np.max(values):.4f}")
                    lines.append(f"  Std Dev: {np.std(values):.4f}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_aggregate_statistics(self) -> AggregateRegimeStats:
        """Get aggregate statistics."""
        return self.aggregate_statistics()
    
    def clear(self):
        """Clear all accumulated statistics."""
        self.day_stats.clear()
        self.classifier.reset()
