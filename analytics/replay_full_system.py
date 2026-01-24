"""
Full-System Replay Driver (Phase v1.1.2).

Loads historical days and runs full engine pipeline with comprehensive reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
import logging
from pathlib import Path
import json

from engine.portfolio_risk_manager import PortfolioRiskManager
from engine.policy_engine import PolicyEngine
from engine.causal_evaluator import CausalEvaluator
from engine.execution_simulator import ExecutionSimulator


class FullSystemReplayDriver:
    """Orchestrates full engine pipeline replay across historical days."""
    
    def __init__(
        self,
        capital: float = 1000000,
        output_dir: str = 'logs/system'
    ):
        """Initialize replay driver."""
        self.capital = capital
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize engine components
        self.portfolio = PortfolioRiskManager(
            total_capital=capital,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000
        )
        
        self.policy_engine = PolicyEngine()
        self.evaluator = CausalEvaluator()
        self.execution_sim = ExecutionSimulator()
        
        # Configure logging
        self.logger = self._setup_logger()
        
        # Results tracking
        self.results = {
            'session_transitions': [],
            'flow_context_stats': {},
            'evaluator_scores': [],
            'policy_decisions': [],
            'execution_slippage': [],
            'risk_scaling': [],
            'capacity_events': [],
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up replay logger."""
        logger = logging.getLogger('FullSystemReplay')
        logger.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f'replay_full_{timestamp}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
        # 480 minutes = 8 hours of trading
        timestamps = pd.date_range(base_date, periods=480, freq='1min')
        
        # Default volatility by session
        if volatility_profile is None:
            volatility_profile = {
                'GLOBEX': 0.0005,
                'PREMARKET': 0.0008,
                'RTH_OPEN': 0.0015,
                'MIDDAY': 0.0008,
                'POWER_HOUR': 0.0012,
                'CLOSE': 0.0010,
            }
        
        # Session definitions
        sessions = {
            'GLOBEX': (0, 60),
            'PREMARKET': (60, 120),
            'RTH_OPEN': (120, 180),
            'MIDDAY': (180, 300),
            'POWER_HOUR': (300, 360),
            'CLOSE': (360, 480),
        }
        
        # Generate prices
        prices = [4500.0]
        
        for i in range(1, len(timestamps)):
            # Determine session
            session_name = 'MIDDAY'
            for sname, (start, end) in sessions.items():
                if start <= i < end:
                    session_name = sname
                    break
            
            vol = volatility_profile.get(session_name, 0.001)
            
            # Apply day type characteristics
            if day_type == 'trend':
                drift = 0.0002  # Slight upward trend
            elif day_type == 'reversal':
                drift = -0.00015 if i < 240 else 0.0005  # Down then up
            elif day_type == 'range':
                drift = 0.0  # No drift
            else:  # normal
                drift = 0.00005
            
            change = drift * prices[-1] + np.random.normal(0, vol * prices[-1])
            new_price = prices[-1] + change
            prices.append(max(new_price, 4300))
        
        # Generate volumes
        volumes = []
        for i, ts in enumerate(timestamps):
            session_name = 'MIDDAY'
            for sname, (start, end) in sessions.items():
                if start <= i < end:
                    session_name = sname
                    break
            
            if session_name in ['RTH_OPEN', 'POWER_HOUR', 'CLOSE']:
                volume = np.random.uniform(8000, 15000)
            elif session_name == 'MIDDAY':
                volume = np.random.uniform(3000, 7000)
            else:
                volume = np.random.uniform(1000, 4000)
            volumes.append(volume)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.003 for p in prices],
            'low': [p * 0.997 for p in prices],
            'close': prices,
            'volume': volumes,
            'session': [self._get_session_name(i) for i in range(len(timestamps))],
            'day_type': day_type,
        })
        
        return df
    
    def _get_session_name(self, minute: int) -> str:
        """Get session name for a given minute."""
        sessions = {
            'GLOBEX': (0, 60),
            'PREMARKET': (60, 120),
            'RTH_OPEN': (120, 180),
            'MIDDAY': (180, 300),
            'POWER_HOUR': (300, 360),
            'CLOSE': (360, 480),
        }
        
        for sname, (start, end) in sessions.items():
            if start <= minute < end:
                return sname
        return 'UNKNOWN'
    
    def replay_day(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Replay a full day of trading through the engine."""
        day_type = df['day_type'].iloc[0] if 'day_type' in df.columns else 'unknown'
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"REPLAYING {day_type.upper()} DAY - {len(df)} minutes")
        self.logger.info(f"{'='*70}")
        
        session_transitions = []
        evaluator_scores = []
        policy_decisions_log = []
        execution_events = []
        current_session = None
        
        # Process each minute
        for i, row in df.iterrows():
            session_name = row['session']
            price = row['close']
            volume = row['volume']
            
            # Track session transitions
            if session_name != current_session:
                transition = {
                    'from_session': current_session,
                    'to_session': session_name,
                    'minute': i,
                    'price': price,
                }
                session_transitions.append(transition)
                self.logger.info(f"[TRANSITION] {current_session} -> {session_name} @ {price:.2f}")
                current_session = session_name
            
            # Create policy decision
            policy_decision = {
                'session_name': session_name,
                'session_modifiers': {},
                'flow_signals': {},
            }
            
            # Create volume state
            volume_state = {
                'volume_1min': volume,
                'volume_5min': df.iloc[max(0, i-4):i+1]['volume'].sum(),
            }
            
            # Evaluate risk
            result = self.portfolio.evaluate_risk_with_context(
                symbol='ES',
                target_size=20,
                price=price,
                policy_decision=policy_decision,
                volume_state=volume_state
            )
            
            evaluator_scores.append({
                'minute': i,
                'session': session_name,
                'action': result.action,
                'approved_size': result.approved_size,
                'confidence': result.confidence,
                'session_factor': result.risk_scaling_factors.get('session_factor', 1.0),
            })
            
            policy_decisions_log.append({
                'minute': i,
                'session': session_name,
                'action': result.action,
                'reasoning': result.reasoning,
            })
            
            if i % 60 == 0:  # Log every hour
                self.logger.debug(
                    f"[{session_name:12}] Min {i:3d} | Price: {price:8.2f} | "
                    f"Action: {result.action:12} | Approved: {result.approved_size:6.2f}"
                )
        
        return {
            'day_type': day_type,
            'minutes_replayed': len(df),
            'session_transitions': session_transitions,
            'evaluator_scores': evaluator_scores,
            'policy_decisions': policy_decisions_log,
        }
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate text summary report."""
        report = []
        report.append("\n" + "="*70)
        report.append("FULL-SYSTEM REPLAY VALIDATION REPORT")
        report.append("="*70)
        
        for day_result in results:
            report.append(f"\nDay Type: {day_result['day_type'].upper()}")
            report.append(f"Minutes Replayed: {day_result['minutes_replayed']}")
            
            # Session transitions
            report.append(f"\nSession Transitions: {len(day_result['session_transitions'])}")
            for trans in day_result['session_transitions']:
                if trans['from_session']:
                    report.append(
                        f"  {trans['from_session']} -> {trans['to_session']} "
                        f"@ min {trans['minute']} (price: {trans['price']:.2f})"
                    )
            
            # Evaluator statistics
            scores = pd.DataFrame(day_result['evaluator_scores'])
            report.append(f"\nEvaluator Statistics:")
            report.append(f"  Mean Confidence: {scores['confidence'].mean():.2%}")
            report.append(f"  Std Confidence: {scores['confidence'].std():.2%}")
            
            # Policy decision distribution
            decisions = scores['action'].value_counts()
            report.append(f"\nPolicy Decision Distribution:")
            for action, count in decisions.items():
                pct = count / len(scores) * 100
                report.append(f"  {action}: {count:3d} ({pct:5.1f}%)")
            
            # Session factor statistics
            session_factors = scores['session_factor']
            report.append(f"\nSession Factor Statistics:")
            report.append(f"  Mean: {session_factors.mean():.4f}")
            report.append(f"  Min: {session_factors.min():.4f}")
            report.append(f"  Max: {session_factors.max():.4f}")
        
        report.append("\n" + "="*70)
        return "\n".join(report)
    
    def run_replay_suite(self):
        """Run replay across multiple day types."""
        day_types = ['normal', 'trend', 'reversal', 'range']
        base_date = datetime(2026, 1, 20, 0, 0, 0)
        
        all_results = []
        
        for day_type in day_types:
            # Create synthetic day
            df = self._create_synthetic_day(
                day_type=day_type,
                base_date=base_date + timedelta(days=len(all_results))
            )
            
            # Replay
            result = self.replay_day(df)
            all_results.append(result)
            
            # Reset portfolio for next day
            self.portfolio = PortfolioRiskManager(
                total_capital=self.capital,
                max_symbol_exposure=500000,
                max_total_exposure=1000000,
                max_daily_loss=50000
            )
        
        # Generate report
        report = self.generate_summary_report(all_results)
        self.logger.info(report)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'replay_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"\nReport saved to: {report_file}")
        
        return all_results


if __name__ == '__main__':
    driver = FullSystemReplayDriver()
    driver.run_replay_suite()
