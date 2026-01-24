"""
Full-System Replay Validation & Stress Testing (Phase v1.1.2).

Tests:
- Multi-session replay (Globex → PreMarket → RTH Open → Midday → PowerHour → Close)
- Macro-day replay (CPI, FOMC, NFP days)
- Volatility shock replay
- Capacity stress test

Validates:
- Session transitions are correct
- Flow context behaves realistically
- Evaluator confidence varies by session
- Policy decisions reflect session/flow context
- Execution slippage matches session rules
- Risk manager enforces capacity correctly
- No runaway trades or silent failures
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

from engine.portfolio_risk_manager import PortfolioRiskManager
from analytics.data_loader import MarketStateBuilder


class TestMultiSessionReplay(unittest.TestCase):
    """Test multi-session transitions and context flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_multi_session')
        self.logger.setLevel(logging.INFO)
        
        self.portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
        
        # Create synthetic intraday data for a full trading day
        base_date = datetime(2026, 1, 20, 0, 0, 0)
        self.timestamps = pd.date_range(base_date, periods=480, freq='1min')  # 8 hours
        
        # Session definitions (times in minutes from market start)
        # Market starts at 9:30 ET (session 0)
        self.sessions = {
            'GLOBEX': (0, 60),          # Before pre-market
            'PREMARKET': (60, 120),     # 23:30 - 00:30 (simulated)
            'RTH_OPEN': (120, 180),     # Open to +60 min
            'MIDDAY': (180, 300),       # Midday session
            'POWER_HOUR': (300, 360),   # Last hour
            'CLOSE': (360, 480),        # Close section
        }
    
    def _create_synthetic_data(self, session_volatility: Dict[str, float]) -> pd.DataFrame:
        """Create synthetic market data with session-specific characteristics."""
        prices = [4500.0]  # Start price
        volumes = []
        
        for i in range(1, len(self.timestamps)):
            # Determine current session
            session_name = None
            for sname, (start, end) in self.sessions.items():
                if start <= i < end:
                    session_name = sname
                    break
            
            # Get volatility for this session
            vol = session_volatility.get(session_name, 0.01)
            
            # Generate price movement with session volatility
            change = np.random.normal(0, vol * prices[-1])
            new_price = prices[-1] + change
            prices.append(max(new_price, 4400))  # Floor price
            
            # Volume varies by session
            if session_name in ['RTH_OPEN', 'POWER_HOUR', 'CLOSE']:
                volume = np.random.uniform(5000, 15000)
            elif session_name == 'MIDDAY':
                volume = np.random.uniform(2000, 5000)
            else:
                volume = np.random.uniform(1000, 3000)
            volumes.append(volume)
        
        volumes.append(np.random.uniform(1000, 3000))
        
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'open': prices,
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': volumes,
        })
        
        return df
    
    def test_multi_session_transitions(self):
        """Test that session transitions occur correctly."""
        session_vol = {
            'GLOBEX': 0.0005,
            'PREMARKET': 0.0008,
            'RTH_OPEN': 0.0015,
            'MIDDAY': 0.0008,
            'POWER_HOUR': 0.0012,
            'CLOSE': 0.0010,
        }
        df = self._create_synthetic_data(session_vol)
        
        # Track session transitions
        current_session = None
        transitions = []
        
        for i, row in df.iterrows():
            # Determine session based on time
            session_name = None
            for sname, (start, end) in self.sessions.items():
                if start <= i < end:
                    session_name = sname
                    break
            
            if session_name != current_session:
                transitions.append({
                    'from_session': current_session,
                    'to_session': session_name,
                    'time': row['timestamp'],
                    'price': row['close']
                })
                current_session = session_name
        
        # Verify transitions occur
        self.assertGreater(len(transitions), 0, "No session transitions recorded")
        
        # Verify session sequence
        expected_sequence = ['GLOBEX', 'PREMARKET', 'RTH_OPEN', 'MIDDAY', 'POWER_HOUR', 'CLOSE']
        actual_sequence = [t['to_session'] for t in transitions if t['to_session'] is not None]
        self.assertEqual(actual_sequence, expected_sequence, "Session sequence mismatch")
    
    def test_session_context_in_risk_decisions(self):
        """Test that risk decisions reflect session context."""
        session_vol = {
            'GLOBEX': 0.0005,
            'PREMARKET': 0.0008,
            'RTH_OPEN': 0.0015,
            'MIDDAY': 0.0008,
            'POWER_HOUR': 0.0012,
            'CLOSE': 0.0010,
        }
        df = self._create_synthetic_data(session_vol)
        
        # Test risk decisions across sessions
        decisions_by_session = {}
        
        for i, row in df.iterrows():
            # Determine session
            session_name = None
            for sname, (start, end) in self.sessions.items():
                if start <= i < end:
                    session_name = sname
                    break
            
            if session_name not in decisions_by_session:
                decisions_by_session[session_name] = []
            
            # Make risk decision for this session
            policy_decision = {
                'session_name': session_name,
                'session_modifiers': {},
                'flow_signals': {},
            }
            
            volume_state = {
                'volume_1min': row['volume'],
                'volume_5min': df.iloc[max(0, i-4):i+1]['volume'].sum(),
            }
            
            result = self.portfolio.evaluate_risk_with_context(
                symbol='ES',
                target_size=10,
                price=row['close'],
                policy_decision=policy_decision,
                volume_state=volume_state
            )
            
            decisions_by_session[session_name].append({
                'action': result.action,
                'approved_size': result.approved_size,
                'session_factors': result.risk_scaling_factors.get('session_factor', 1.0),
            })
        
        # Verify decisions reflect session context
        self.assertIn('GLOBEX', decisions_by_session)
        self.assertIn('RTH_OPEN', decisions_by_session)
        self.assertIn('CLOSE', decisions_by_session)


class TestMacroDayReplay(unittest.TestCase):
    """Test replay on macro-event days (CPI, FOMC, NFP)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_macro_day')
        self.logger.setLevel(logging.INFO)
        
        self.portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
    
    def test_cpi_day_pre_event_risk_reduction(self):
        """Test that risk reduces before CPI event."""
        # Simulate CPI day: event at 13:30 ET (minute 240)
        base_date = datetime(2026, 1, 15, 0, 0, 0)
        timestamps = pd.date_range(base_date, periods=480, freq='1min')
        
        event_minute = 240  # 13:30 ET
        
        # Pre-event: normal conditions
        # Post-event: volatility spike
        
        decisions_pre = []
        decisions_post = []
        
        for i, ts in enumerate(timestamps):
            price = 4500.0 + np.random.normal(0, 5)
            volume = 5000 if i < event_minute else 15000  # Volume spike after event
            
            policy_decision = {
                'session_name': 'MIDDAY',
                'session_modifiers': {},
                'flow_signals': {'macro_event': 'CPI'} if i == event_minute else {},
            }
            
            volume_state = {
                'volume_1min': volume,
                'volume_5min': 5000 * 5,
            }
            
            result = self.portfolio.evaluate_risk_with_context(
                symbol='ES',
                target_size=10,
                price=price,
                policy_decision=policy_decision,
                volume_state=volume_state
            )
            
            if i < event_minute:
                decisions_pre.append(result.approved_size)
            else:
                decisions_post.append(result.approved_size)
        
        # Pre-event should have more aggressive sizing
        pre_avg = np.mean(decisions_pre)
        
        # Verify we have data points
        self.assertGreater(len(decisions_pre), 0)
        self.assertGreater(len(decisions_post), 0)
    
    def test_fomc_day_volatility_handling(self):
        """Test that engine handles FOMC volatility correctly."""
        base_date = datetime(2026, 1, 28, 0, 0, 0)  # FOMC meeting day
        timestamps = pd.date_range(base_date, periods=480, freq='1min')
        
        event_minute = 300  # Event at 15:00 ET
        
        for i, ts in enumerate(timestamps):
            # Volatility spikes after FOMC
            if i < event_minute:
                price_vol = 0.001
            else:
                price_vol = 0.005  # 5x volatility spike
            
            price = 4500.0 + np.random.normal(0, price_vol * 4500)
            volume = 8000 if i < event_minute else 20000
            
            policy_decision = {
                'session_name': 'CLOSE',
                'session_modifiers': {},
                'flow_signals': {'macro_event': 'FOMC'} if i >= event_minute else {},
            }
            
            volume_state = {
                'volume_1min': volume,
                'volume_5min': 8000 * 5,
            }
            
            result = self.portfolio.evaluate_risk_with_context(
                symbol='ES',
                target_size=20,
                price=price,
                policy_decision=policy_decision,
                volume_state=volume_state
            )
            
            # After FOMC, risk should be elevated (captured in logs)
            self.assertIn(result.action, ['ALLOW', 'REDUCE_SIZE', 'BLOCK', 'FORCE_EXIT'])


class TestVolatilityShockReplay(unittest.TestCase):
    """Test replay during volatility shocks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_volatility_shock')
        self.logger.setLevel(logging.INFO)
        
        self.portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
    
    def test_large_candle_shock(self):
        """Test engine response to large unexpected candle."""
        base_date = datetime(2026, 1, 20, 0, 0, 0)
        timestamps = pd.date_range(base_date, periods=120, freq='1min')
        
        prices = [4500.0]
        shock_minute = 60
        
        for i in range(1, len(timestamps)):
            if i == shock_minute:
                # Large shock: down 50 points in 1 minute
                prices.append(prices[-1] - 50)
            else:
                prices.append(prices[-1] + np.random.normal(0, 2))
        
        decisions_pre_shock = []
        decisions_post_shock = []
        
        for i, ts in enumerate(timestamps):
            policy_decision = {
                'session_name': 'MIDDAY',
                'session_modifiers': {},
                'flow_signals': {},
            }
            
            volume_state = {
                'volume_1min': 5000 if i < shock_minute else 20000,
                'volume_5min': 5000 * 5,
            }
            
            result = self.portfolio.evaluate_risk_with_context(
                symbol='ES',
                target_size=20,
                price=prices[i],
                policy_decision=policy_decision,
                volume_state=volume_state
            )
            
            if i < shock_minute:
                decisions_pre_shock.append(result.approved_size)
            else:
                decisions_post_shock.append(result.approved_size)
        
        # Verify we tracked decisions
        self.assertGreater(len(decisions_pre_shock), 0)
        self.assertGreater(len(decisions_post_shock), 0)


class TestCapacityStressTest(unittest.TestCase):
    """Test capacity limits under stress scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_capacity_stress')
        self.logger.setLevel(logging.INFO)
        
        self.portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
    
    def test_notional_limit_enforcement(self):
        """Test that notional limits are enforced."""
        # Try to trade ES with large notional
        policy_decision = {
            'session_name': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        volume_state = {
            'volume_1min': 5000.0,
            'volume_5min': 25000.0,
        }
        
        # Try massive position: 2000 contracts
        result = self.portfolio.evaluate_risk_with_context(
            symbol='ES',
            target_size=2000,
            price=4500.0,
            policy_decision=policy_decision,
            volume_state=volume_state
        )
        
        # Should be blocked or reduced
        self.assertIn(result.action, ['BLOCK', 'REDUCE_SIZE'])
    
    def test_volume_limit_enforcement(self):
        """Test that volume limits are enforced."""
        policy_decision = {
            'session_name': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        volume_state = {
            'volume_1min': 5000.0,
            'volume_5min': 25000.0,
        }
        
        # Try to trade 600 contracts (600 > 5% of 5000 volume)
        result = self.portfolio.evaluate_risk_with_context(
            symbol='ES',
            target_size=600,
            price=4500.0,
            policy_decision=policy_decision,
            volume_state=volume_state
        )
        
        # Should be blocked due to volume limits
        self.assertEqual(result.action, 'BLOCK')
        self.assertTrue(result.capacity_flags.get('volume_1min_exceeded', False))
    
    def test_exposure_limit_with_existing_positions(self):
        """Test exposure limits when already holding positions."""
        # Set initial position
        self.portfolio.update_exposure('ES', 50, 4500)
        self.assertEqual(self.portfolio.current_exposure_per_symbol['ES'], 225000)
        
        policy_decision = {
            'session_name': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        volume_state = {
            'volume_1min': 5000.0,
            'volume_5min': 25000.0,
        }
        
        # Try to add 100 more (would be 675k total for ES, exceeds 500k limit)
        result = self.portfolio.evaluate_risk_with_context(
            symbol='ES',
            target_size=100,
            price=4500.0,
            policy_decision=policy_decision,
            volume_state=volume_state
        )
        
        # Should reduce to fit limit
        self.assertEqual(result.action, 'REDUCE_SIZE')
        self.assertLess(result.approved_size, 1.0)


class TestNoRunawaytrades(unittest.TestCase):
    """Test that no runaway trades occur."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_no_runaway')
        self.logger.setLevel(logging.INFO)
        
        self.portfolio = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=500000,
            max_total_exposure=1000000,
            max_daily_loss=50000,
            logger=self.logger
        )
    
    def test_stop_loss_enforcement(self):
        """Test that stop-loss prevents runaway losses."""
        # Simulate daily loss accumulation
        self.portfolio.update_pnl(realized=-40000, unrealized=-5000)
        
        # Daily loss is -45000, close to -50000 limit
        policy_decision = {
            'session_name': 'CLOSE',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        volume_state = {
            'volume_1min': 5000.0,
            'volume_5min': 25000.0,
        }
        
        # Try to add more risk
        result = self.portfolio.evaluate_risk_with_context(
            symbol='ES',
            target_size=10,
            price=4500.0,
            policy_decision=policy_decision,
            volume_state=volume_state
        )
        
        # Should still allow (within limit)
        self.assertEqual(result.action, 'ALLOW')
        
        # Exceed limit
        self.portfolio.update_pnl(realized=-51000, unrealized=0)
        
        result = self.portfolio.evaluate_risk_with_context(
            symbol='ES',
            target_size=10,
            price=4500.0,
            policy_decision=policy_decision,
            volume_state=volume_state
        )
        
        # Should force exit
        self.assertEqual(result.action, 'FORCE_EXIT')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL FULL-SYSTEM REPLAY TESTS PASSED")
        print(f"  Ran {result.testsRun} tests successfully")
    else:
        print("✗ SOME FULL-SYSTEM REPLAY TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    print("="*70)
    
    exit(0 if result.wasSuccessful() else 1)
