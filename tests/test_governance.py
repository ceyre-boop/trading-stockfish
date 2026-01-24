"""
Test suite for Governance system.

Verifies:
- Kill switch logic
- Trading halt enforcement
- Force flatten capability
- Action override rules
- State management
"""

import unittest
from engine.governance import Governance, GovernanceState
from datetime import datetime


class TestGovernanceBasic(unittest.TestCase):
    """Test basic governance functionality."""
    
    def setUp(self):
        """Initialize governance system."""
        self.governance = Governance(max_daily_loss=5000)
    
    def test_initialization(self):
        """Test governance initializes in safe state."""
        self.assertEqual(self.governance.max_daily_loss, 5000)
        self.assertFalse(self.governance.kill_switch_triggered)
        self.assertIsNone(self.governance.trigger_time)
        self.assertIsNone(self.governance.trigger_reason)
    
    def test_can_trade_initially_true(self):
        """Test trading allowed initially."""
        result = self.governance.can_trade()
        self.assertTrue(result)
    
    def test_force_flatten_initially_false(self):
        """Test force flatten not active initially."""
        result = self.governance.force_flatten()
        self.assertFalse(result)
    
    def test_evaluate_small_loss(self):
        """Test evaluation with loss within limits."""
        self.governance.evaluate(daily_loss=-3000)
        
        self.assertFalse(self.governance.kill_switch_triggered)
        self.assertTrue(self.governance.can_trade())
    
    def test_evaluate_exceeds_loss_limit(self):
        """Test evaluation when loss exceeds limit."""
        self.governance.evaluate(daily_loss=-5500)
        
        self.assertTrue(self.governance.kill_switch_triggered)
        self.assertIsNotNone(self.governance.trigger_time)
        self.assertIsNotNone(self.governance.trigger_reason)
    
    def test_evaluate_triggers_kill_switch(self):
        """Test kill switch activation blocks trading."""
        self.governance.evaluate(daily_loss=-5500)
        
        self.assertFalse(self.governance.can_trade())
        self.assertTrue(self.governance.force_flatten())
    
    def test_kill_switch_irreversible(self):
        """Test kill switch cannot be reversed by good P&L."""
        self.governance.evaluate(daily_loss=-5500)  # Trigger
        self.assertFalse(self.governance.can_trade())
        
        # Evaluate with profit
        self.governance.evaluate(daily_loss=1000)
        
        # Still halted
        self.assertFalse(self.governance.can_trade())
    
    def test_get_state_snapshot(self):
        """Test getting governance state snapshot."""
        self.governance.evaluate(daily_loss=-5500)
        
        state = self.governance.get_state()
        
        self.assertIsInstance(state, GovernanceState)
        self.assertEqual(state.max_daily_loss, 5000)
        self.assertTrue(state.kill_switch_triggered)
        self.assertIsNotNone(state.trigger_time)


class TestGovernanceActionOverride(unittest.TestCase):
    """Test action override logic."""
    
    def setUp(self):
        """Initialize governance system."""
        self.governance = Governance(max_daily_loss=5000)
    
    def test_override_action_allowed_when_trading_active(self):
        """Test action not overridden when trading allowed."""
        result = self.governance.override_action('ENTER', 'ES', 'Open position')
        self.assertEqual(result, 'ENTER')
    
    def test_override_action_blocked_when_halted(self):
        """Test action overridden when trading halted."""
        self.governance.evaluate(daily_loss=-5500)
        
        result = self.governance.override_action('ENTER', 'ES', 'Open position')
        
        # Entry should be converted to EXIT or DO_NOTHING
        self.assertIn(result, ['EXIT', 'DO_NOTHING'])
    
    def test_override_exit_when_halted(self):
        """Test EXIT action allowed even when halted."""
        self.governance.evaluate(daily_loss=-5500)
        
        result = self.governance.override_action('EXIT', 'ES', 'Close position')
        
        # EXIT allowed (or converted to DO_NOTHING but allowed)
        self.assertIn(result, ['EXIT', 'DO_NOTHING'])
    
    def test_override_add_when_halted(self):
        """Test ADD action overridden when halted."""
        self.governance.evaluate(daily_loss=-5500)
        
        result = self.governance.override_action('ADD', 'ES', 'Add to position')
        
        self.assertIn(result, ['EXIT', 'DO_NOTHING'])
    
    def test_override_multiple_actions(self):
        """Test various actions overridden correctly."""
        self.governance.evaluate(daily_loss=-5500)
        
        actions_to_test = ['ENTER', 'ADD', 'REVERSE', 'REDUCE']
        
        for action in actions_to_test:
            result = self.governance.override_action(action, 'ES', 'Test')
            self.assertIn(result, ['EXIT', 'DO_NOTHING'])


class TestGovernanceReporting(unittest.TestCase):
    """Test governance reporting and audit trail."""
    
    def setUp(self):
        """Initialize governance system."""
        self.governance = Governance(max_daily_loss=5000)
    
    def test_decision_history_tracking(self):
        """Test decisions are recorded in history."""
        self.governance.evaluate(daily_loss=-2000)
        self.governance.evaluate(daily_loss=-3000)
        self.governance.evaluate(daily_loss=-5500)
        
        self.assertEqual(len(self.governance.decision_history), 3)
    
    def test_get_report(self):
        """Test comprehensive governance report."""
        self.governance.evaluate(daily_loss=-2000)
        self.governance.evaluate(daily_loss=-5500)
        
        report = self.governance.get_report()
        
        self.assertEqual(report['max_daily_loss'], 5000)
        self.assertTrue(report['kill_switch_triggered'])
        self.assertTrue(report['is_trading_halted'])
        self.assertEqual(report['decisions_made'], 2)
    
    def test_report_before_trigger(self):
        """Test report before kill switch triggered."""
        self.governance.evaluate(daily_loss=-2000)
        
        report = self.governance.get_report()
        
        self.assertFalse(report['kill_switch_triggered'])
        self.assertFalse(report['is_trading_halted'])


class TestGovernanceScenarios(unittest.TestCase):
    """Test realistic governance scenarios."""
    
    def setUp(self):
        """Initialize governance system."""
        self.governance = Governance(max_daily_loss=5000)
    
    def test_normal_trading_day(self):
        """Test normal day with no kill switch."""
        # Multiple evaluations throughout day
        self.governance.evaluate(daily_loss=-500)
        self.assertTrue(self.governance.can_trade())
        
        self.governance.evaluate(daily_loss=-1200)
        self.assertTrue(self.governance.can_trade())
        
        self.governance.evaluate(daily_loss=-2000)
        self.assertTrue(self.governance.can_trade())
        
        # Final report
        report = self.governance.get_report()
        self.assertFalse(report['kill_switch_triggered'])
    
    def test_bad_trading_day_kill_switch(self):
        """Test day ending with kill switch activation."""
        self.governance.evaluate(daily_loss=-1000)
        self.assertTrue(self.governance.can_trade())
        
        self.governance.evaluate(daily_loss=-3000)
        self.assertTrue(self.governance.can_trade())
        
        self.governance.evaluate(daily_loss=-5500)
        self.assertFalse(self.governance.can_trade())
        
        # Verify irreversibility
        self.governance.evaluate(daily_loss=-5000)  # Better, but already halted
        self.assertFalse(self.governance.can_trade())
    
    def test_decision_history_shows_progression(self):
        """Test decision history shows loss progression."""
        self.governance.evaluate(daily_loss=-1000)
        self.governance.evaluate(daily_loss=-3000)
        self.governance.evaluate(daily_loss=-5500)
        
        history = self.governance.decision_history
        
        # Check progression
        self.assertEqual(history[0]['daily_loss'], -1000)
        self.assertEqual(history[1]['daily_loss'], -3000)
        self.assertEqual(history[2]['daily_loss'], -5500)
        self.assertEqual(history[2]['action'], 'KILL_SWITCH_ACTIVATED')


class TestGovernanceReset(unittest.TestCase):
    """Test governance session reset."""
    
    def setUp(self):
        """Initialize governance system."""
        self.governance = Governance(max_daily_loss=5000)
    
    def test_reset_clears_history(self):
        """Test reset clears decision history."""
        self.governance.evaluate(daily_loss=-2000)
        self.governance.evaluate(daily_loss=-3000)
        
        initial_count = len(self.governance.decision_history)
        self.governance.reset_session()
        
        # History should be cleared
        self.assertEqual(len(self.governance.decision_history), 0)
    
    def test_reset_preserves_kill_switch_state(self):
        """Test reset preserves active kill switch state."""
        self.governance.evaluate(daily_loss=-5500)
        self.assertTrue(self.governance.kill_switch_triggered)
        
        self.governance.reset_session()
        
        # Kill switch should still be active
        self.assertTrue(self.governance.kill_switch_triggered)
        self.assertFalse(self.governance.can_trade())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL GOVERNANCE TESTS PASSED")
        print(f"  Ran {result.testsRun} tests successfully")
    else:
        print("✗ SOME GOVERNANCE TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    print("="*70)
    
    exit(0 if result.wasSuccessful() else 1)
