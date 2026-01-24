"""
Regime Preview Replay (Phase v1.2).

Replay full trading days showing regime classification in real-time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from engine.regime_classifier import RegimeClassifier


class RegimePreviewReplay:
    """Replay trading days with regime classification tracking."""
    
    def __init__(self, output_dir: str = 'logs/system'):
        """Initialize regime preview replay."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.classifier = RegimeClassifier(logger=self.logger)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger('RegimePreview')
        logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f'regime_preview_{timestamp}.log'
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _create_synthetic_day(
        self,
        day_type: str,
        base_date: datetime,
        volatility_profile: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Create synthetic market data for a specific day type."""
        timestamps = pd.date_range(base_date, periods=480, freq='1min')
        
        prices = [4500.0]
        
        for i in range(1, len(timestamps)):
            if day_type == 'trend':
                # Strong uptrend with occasional pullbacks
                drift = 0.0003
                noise = np.random.normal(0, 3)
                vol_spike = 10 if i % 60 == 0 else 0  # Volatility spike every hour
                price = prices[-1] + (drift * prices[-1]) + noise + vol_spike
            
            elif day_type == 'reversal':
                # Up first half, reversal second half
                if i < 240:
                    drift = 0.0004
                else:
                    drift = -0.0005
                noise = np.random.normal(0, 3)
                price = prices[-1] + (drift * prices[-1]) + noise
            
            elif day_type == 'range':
                # Oscillation around center
                center = 4500.0
                sin_val = np.sin(i * np.pi / 120)
                price = center + (sin_val * 30) + np.random.normal(0, 2)
            
            else:  # normal
                # Random walk
                price = prices[-1] + np.random.normal(0, 3)
            
            prices.append(max(price, 4300))
        
        # Generate volumes
        volumes = []
        for i in range(len(timestamps)):
            # Higher volume at open and close
            if i < 60 or i > 420:
                volume = np.random.uniform(8000, 15000)
            else:
                volume = np.random.uniform(3000, 7000)
            volumes.append(volume)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.003 for p in prices],
            'low': [p * 0.997 for p in prices],
            'close': prices,
            'volume': volumes,
        })
        
        # Calculate VWAP
        df['vwap'] = df['close']  # Simplified VWAP
        
        return df
    
    def replay_day_with_regime(self, df: pd.DataFrame, day_type: str) -> Dict[str, Any]:
        """Replay day tracking regime classification."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"REGIME PREVIEW REPLAY - {day_type.upper()} DAY")
        self.logger.info(f"{'='*70}")
        
        regime_transitions = []
        regime_history = []
        current_regime = None
        
        # Track every 30 minutes for summary
        summary_intervals = []
        
        for i, row in df.iterrows():
            price = row['close']
            high = row['high']
            low = row['low']
            volume = row['volume']
            vwap = row['vwap']
            
            # Generate synthetic flow signals (less frequent)
            flow_signals = {
                'initiative_move_detected': i % 15 == 0,
                'stop_run_detected': day_type == 'reversal' and i % 20 == 0,
            }
            
            # Update regime
            regime = self.classifier.update_with_bar(
                timestamp=row['timestamp'],
                open_price=price,
                high=high,
                low=low,
                close=price,
                volume=volume,
                vwap=vwap,
                flow_signals=flow_signals
            )
            
            regime_history.append({
                'minute': i,
                'price': price,
                'regime': regime.regime_label,
                'confidence': regime.regime_confidence,
            })
            
            # Track transitions
            if regime.regime_label != current_regime:
                transition = {
                    'from_regime': current_regime,
                    'to_regime': regime.regime_label,
                    'minute': i,
                    'price': price,
                    'confidence': regime.regime_confidence,
                }
                regime_transitions.append(transition)
                
                self.logger.info(
                    f"[{i:3d}] REGIME TRANSITION: {current_regime} -> {regime.regime_label} "
                    f"(confidence: {regime.regime_confidence:.0%})"
                )
                
                current_regime = regime.regime_label
            
            # Summary every 30 minutes
            if (i + 1) % 30 == 0:
                signals_summary = ', '.join(
                    [f"{s.name}={s.value:.2f}" for s in regime.contributing_signals[:3]]
                )
                self.logger.debug(
                    f"[{i:3d}] {regime.regime_label:10} ({regime.regime_confidence:.0%}) | {signals_summary}"
                )
                
                summary_intervals.append({
                    'minute': i,
                    'regime': regime.regime_label,
                    'confidence': regime.regime_confidence,
                    'signals': regime.contributing_signals,
                    'features': regime.regime_features,
                })
        
        return {
            'day_type': day_type,
            'regime_transitions': regime_transitions,
            'regime_history': regime_history,
            'summary_intervals': summary_intervals,
        }
    
    def generate_report(self, day_results: List[Dict[str, Any]]) -> str:
        """Generate regime preview report."""
        report = []
        report.append("\n" + "="*70)
        report.append("REGIME CLASSIFICATION PREVIEW REPORT")
        report.append("="*70)
        
        for result in day_results:
            day_type = result['day_type'].upper()
            report.append(f"\n{day_type} DAY")
            report.append("-" * 70)
            
            # Transitions
            transitions = result['regime_transitions']
            report.append(f"\nRegime Transitions: {len(transitions)}")
            for trans in transitions:
                if trans['from_regime']:
                    report.append(
                        f"  Min {trans['minute']:3d}: {trans['from_regime']:10} -> {trans['to_regime']:10} "
                        f"@ {trans['price']:.2f} ({trans['confidence']:.0%})"
                    )
            
            # Distribution
            history = pd.DataFrame(result['regime_history'])
            regime_dist = history['regime'].value_counts()
            
            report.append(f"\nRegime Distribution:")
            for regime, count in regime_dist.items():
                pct = count / len(history) * 100
                report.append(f"  {regime:10} {count:3d} min ({pct:5.1f}%)")
            
            # Confidence statistics
            report.append(f"\nConfidence Statistics:")
            conf_stats = history['confidence'].describe()
            report.append(f"  Mean:     {conf_stats['mean']:.0%}")
            report.append(f"  Min:      {conf_stats['min']:.0%}")
            report.append(f"  Max:      {conf_stats['max']:.0%}")
            report.append(f"  Std Dev:  {conf_stats['std']:.0%}")
            
            # Key insights
            report.append(f"\nKey Insights:")
            
            # Most common regime
            most_common = regime_dist.idxmax()
            most_common_pct = regime_dist.max() / len(history) * 100
            report.append(f"  Dominant regime: {most_common} ({most_common_pct:.1f}%)")
            
            # Regime changes
            regime_changes = sum(1 for trans in transitions if trans['from_regime'] is not None)
            report.append(f"  Regime changes: {regime_changes}")
            
            # Final regime
            final_regime = history.iloc[-1]['regime']
            final_conf = history.iloc[-1]['confidence']
            report.append(f"  Final regime: {final_regime} ({final_conf:.0%})")
        
        report.append("\n" + "="*70)
        return "\n".join(report)
    
    def run_preview_suite(self):
        """Run regime preview across multiple day types."""
        day_types = ['normal', 'trend', 'reversal', 'range']
        base_date = datetime(2026, 1, 20, 0, 0, 0)
        
        all_results = []
        
        for day_type in day_types:
            self.logger.info(f"\nGenerating synthetic {day_type} day...")
            df = self._create_synthetic_day(day_type, base_date + timedelta(days=len(all_results)))
            
            self.logger.info(f"Replaying {day_type} day with regime classification...")
            result = self.replay_day_with_regime(df, day_type)
            all_results.append(result)
            
            # Reset for next day
            self.classifier.reset_session()
        
        # Generate report
        report = self.generate_report(all_results)
        self.logger.info(report)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'regime_preview_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"\nReport saved to: {report_file}")
        
        return all_results


if __name__ == '__main__':
    replay = RegimePreviewReplay()
    replay.run_preview_suite()
