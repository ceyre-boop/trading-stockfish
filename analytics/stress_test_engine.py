"""
Stress Test Harness (Phase v1.1.2).

Tests:
- Volatility stress (artificially inflated volatility)
- Liquidity stress (reduced depth)
- Flow stress (stop-run sequences)
- Capacity stress (large position requests)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

from engine.portfolio_risk_manager import PortfolioRiskManager
from engine.execution_simulator import ExecutionSimulator


class StressTestHarness:
    """Stress test engine under adverse conditions."""
    
    def __init__(self, output_dir: str = 'logs/system'):
        """Initialize stress test harness."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.results = []
    
    def _setup_logger(self) -> logging.Logger:
        """Set up stress test logger."""
        logger = logging.getLogger('StressTest')
        logger.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f'stress_test_{timestamp}.log'
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
    
    def stress_volatility(self, base_volatility: float = 0.001) -> Dict[str, Any]:
        """Test engine under elevated volatility."""
        self.logger.info("\n" + "="*70)
        self.logger.info("VOLATILITY STRESS TEST")
        self.logger.info("="*70)
        
        portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
        
        volatility_multipliers = [1.0, 2.0, 3.0, 5.0, 10.0]
        results = {
            'test_name': 'VOLATILITY_STRESS',
            'results_by_multiplier': []
        }
        
        for mult in volatility_multipliers:
            vol = base_volatility * mult
            
            self.logger.info(f"\nTesting at {mult:.1f}x volatility ({vol:.4f})")
            
            decisions = []
            prices = [4500.0]
            
            # Generate 100 minutes of volatile data
            for i in range(100):
                change = np.random.normal(0, vol * prices[-1])
                prices.append(prices[-1] + change)
                
                policy_decision = {
                    'session_name': 'MIDDAY',
                    'session_modifiers': {},
                    'flow_signals': {},
                }
                
                volume_state = {
                    'volume_1min': 5000.0,
                    'volume_5min': 25000.0,
                }
                
                result = portfolio.evaluate_risk_with_context(
                    symbol='ES',
                    target_size=20,
                    price=prices[-1],
                    policy_decision=policy_decision,
                    volume_state=volume_state
                )
                
                decisions.append({
                    'multiplier': mult,
                    'action': result.action,
                    'approved_size': result.approved_size,
                    'confidence': result.confidence,
                })
            
            # Analyze results
            decisions_df = pd.DataFrame(decisions)
            blocks = len(decisions_df[decisions_df['action'] == 'BLOCK'])
            reduces = len(decisions_df[decisions_df['action'] == 'REDUCE_SIZE'])
            mean_conf = decisions_df['confidence'].mean()
            
            self.logger.info(
                f"  Blocks: {blocks} | Reduces: {reduces} | "
                f"Mean Confidence: {mean_conf:.2%}"
            )
            
            results['results_by_multiplier'].append({
                'multiplier': mult,
                'volatility': vol,
                'blocks': blocks,
                'reduces': reduces,
                'mean_confidence': mean_conf,
            })
        
        return results
    
    def stress_liquidity(self) -> Dict[str, Any]:
        """Test engine under reduced liquidity."""
        self.logger.info("\n" + "="*70)
        self.logger.info("LIQUIDITY STRESS TEST")
        self.logger.info("="*70)
        
        portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
        
        # Test with progressively lower volumes
        volume_levels = [50000, 25000, 10000, 5000, 1000]
        results = {
            'test_name': 'LIQUIDITY_STRESS',
            'results_by_volume': []
        }
        
        for base_volume in volume_levels:
            self.logger.info(f"\nTesting at {base_volume} 1m-volume")
            
            decisions = []
            
            for i in range(100):
                price = 4500.0 + np.random.normal(0, 5)
                volume = base_volume + np.random.uniform(-base_volume*0.2, base_volume*0.2)
                
                policy_decision = {
                    'session_name': 'MIDDAY',
                    'session_modifiers': {},
                    'flow_signals': {},
                }
                
                volume_state = {
                    'volume_1min': max(volume, 100),
                    'volume_5min': volume * 5,
                }
                
                result = portfolio.evaluate_risk_with_context(
                    symbol='ES',
                    target_size=20,
                    price=price,
                    policy_decision=policy_decision,
                    volume_state=volume_state
                )
                
                decisions.append({
                    'volume': base_volume,
                    'action': result.action,
                    'approved_size': result.approved_size,
                })
            
            # Analyze results
            decisions_df = pd.DataFrame(decisions)
            blocks = len(decisions_df[decisions_df['action'] == 'BLOCK'])
            allows = len(decisions_df[decisions_df['action'] == 'ALLOW'])
            
            self.logger.info(f"  Allows: {allows} | Blocks: {blocks}")
            
            results['results_by_volume'].append({
                'volume': base_volume,
                'allows': allows,
                'blocks': blocks,
            })
        
        return results
    
    def stress_flow(self) -> Dict[str, Any]:
        """Test engine under adverse flow (stop-run sequences)."""
        self.logger.info("\n" + "="*70)
        self.logger.info("FLOW STRESS TEST")
        self.logger.info("="*70)
        
        portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
        
        results = {
            'test_name': 'FLOW_STRESS',
            'stop_run_detection': []
        }
        
        self.logger.info("\nSimulating stop-run sequence (repeated rejections)")
        
        decisions = []
        
        for i in range(100):
            price = 4500.0 + np.random.normal(0, 5)
            
            # Simulate stop-run: detect failed attempts
            stop_run_detected = (i % 10 < 5)  # Stop-run 50% of time
            
            policy_decision = {
                'session_name': 'MIDDAY' if i < 50 else 'POWER_HOUR',
                'session_modifiers': {},
                'flow_signals': {
                    'stop_run_detected': stop_run_detected
                },
            }
            
            volume_state = {
                'volume_1min': 5000.0,
                'volume_5min': 25000.0,
            }
            
            result = portfolio.evaluate_risk_with_context(
                symbol='ES',
                target_size=20,
                price=price,
                policy_decision=policy_decision,
                volume_state=volume_state
            )
            
            decisions.append({
                'minute': i,
                'stop_run': stop_run_detected,
                'action': result.action,
                'approved_size': result.approved_size,
            })
        
        # Analyze results
        decisions_df = pd.DataFrame(decisions)
        
        # Check if stop-run reduces sizing
        with_stop_run = decisions_df[decisions_df['stop_run'] == True]
        without_stop_run = decisions_df[decisions_df['stop_run'] == False]
        
        mean_size_with = with_stop_run['approved_size'].mean()
        mean_size_without = without_stop_run['approved_size'].mean()
        
        self.logger.info(
            f"  Mean size with stop-run: {mean_size_with:.4f}\n"
            f"  Mean size without stop-run: {mean_size_without:.4f}"
        )
        
        results['stop_run_detection'].append({
            'mean_size_with_stop_run': mean_size_with,
            'mean_size_without_stop_run': mean_size_without,
            'size_reduction_pct': (1 - mean_size_with/max(mean_size_without, 0.001)) * 100,
        })
        
        return results
    
    def stress_capacity(self) -> Dict[str, Any]:
        """Test engine under capacity stress."""
        self.logger.info("\n" + "="*70)
        self.logger.info("CAPACITY STRESS TEST")
        self.logger.info("="*70)
        
        portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
        
        # Test progressively larger position requests
        size_requests = [10, 50, 100, 200, 500, 1000]
        results = {
            'test_name': 'CAPACITY_STRESS',
            'results_by_size': []
        }
        
        for target_size in size_requests:
            self.logger.info(f"\nTesting {target_size} contract request")
            
            policy_decision = {
                'session_name': 'MIDDAY',
                'session_modifiers': {},
                'flow_signals': {},
            }
            
            volume_state = {
                'volume_1min': 5000.0,
                'volume_5min': 25000.0,
            }
            
            result = portfolio.evaluate_risk_with_context(
                symbol='ES',
                target_size=target_size,
                price=4500.0,
                policy_decision=policy_decision,
                volume_state=volume_state
            )
            
            notional = target_size * 4500.0
            approved_notional = result.approved_size * target_size * 4500.0
            
            self.logger.info(
                f"  Request: {target_size} contracts (${notional:,.0f})\n"
                f"  Action: {result.action}\n"
                f"  Approved: {result.approved_size:.4f}x "
                f"(${approved_notional:,.0f})"
            )
            
            results['results_by_size'].append({
                'target_size': target_size,
                'target_notional': notional,
                'action': result.action,
                'approved_size': result.approved_size,
                'approved_notional': approved_notional,
            })
        
        return results
    
    def run_all_stress_tests(self) -> Dict[str, Any]:
        """Run complete stress test suite."""
        self.logger.info("\n" + "="*70)
        self.logger.info("STARTING FULL STRESS TEST SUITE")
        self.logger.info("="*70)
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
        # Run each stress test
        vol_result = self.stress_volatility()
        all_results['tests'].append(vol_result)
        
        liq_result = self.stress_liquidity()
        all_results['tests'].append(liq_result)
        
        flow_result = self.stress_flow()
        all_results['tests'].append(flow_result)
        
        cap_result = self.stress_capacity()
        all_results['tests'].append(cap_result)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print stress test summary."""
        self.logger.info("\n" + "="*70)
        self.logger.info("STRESS TEST SUMMARY")
        self.logger.info("="*70)
        
        for test in results['tests']:
            self.logger.info(f"\n{test['test_name']}:")
            
            if test['test_name'] == 'VOLATILITY_STRESS':
                for res in test['results_by_multiplier']:
                    self.logger.info(
                        f"  {res['multiplier']:.1f}x: "
                        f"Blocks={res['blocks']}, Reduces={res['reduces']}, "
                        f"Conf={res['mean_confidence']:.2%}"
                    )
            
            elif test['test_name'] == 'LIQUIDITY_STRESS':
                for res in test['results_by_volume']:
                    self.logger.info(
                        f"  Vol={res['volume']}: "
                        f"Allows={res['allows']}, Blocks={res['blocks']}"
                    )
            
            elif test['test_name'] == 'FLOW_STRESS':
                for res in test['stop_run_detection']:
                    self.logger.info(
                        f"  Stop-run reduces sizing by {res['size_reduction_pct']:.1f}%"
                    )
            
            elif test['test_name'] == 'CAPACITY_STRESS':
                for res in test['results_by_size']:
                    self.logger.info(
                        f"  {res['target_size']:4d} contracts: "
                        f"{res['action']:12} "
                        f"(Approved: {res['approved_size']:.4f}x)"
                    )
        
        self.logger.info("\n" + "="*70)
        self.logger.info("STRESS TEST SUITE COMPLETE")
        self.logger.info("="*70)


if __name__ == '__main__':
    harness = StressTestHarness()
    harness.run_all_stress_tests()
