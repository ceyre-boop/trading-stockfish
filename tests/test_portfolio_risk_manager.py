"""
Test suite for PortfolioRiskManager.

Verifies:
- Exposure tracking (per-symbol and total)
- Position sizing constraints
- Daily P&L limits
- Capital utilization
- State snapshots
"""

import unittest
from engine.portfolio_risk_manager import PortfolioRiskManager, PortfolioState
import logging


class TestPortfolioRiskManagerBasic(unittest.TestCase):
    """Test basic portfolio functionality."""
    
    def setUp(self):
        """Initialize portfolio manager."""
        self.portfolio = PortfolioRiskManager(
            total_capital=100000,
            max_symbol_exposure=25000,
            max_total_exposure=50000,
            max_daily_loss=5000
        )
    
    def test_initialization(self):
        """Test portfolio initializes with correct parameters."""
        self.assertEqual(self.portfolio.total_capital, 100000)
        self.assertEqual(self.portfolio.max_symbol_exposure, 25000)
        self.assertEqual(self.portfolio.max_total_exposure, 50000)
        self.assertEqual(self.portfolio.max_daily_loss, 5000)
        self.assertEqual(self.portfolio.current_total_exposure, 0.0)
        self.assertEqual(self.portfolio.daily_pnl, 0.0)
    
    def test_update_exposure_single_symbol(self):
        """Test exposure update for single symbol."""
        self.portfolio.update_exposure('ES', 10, 4500)
        
        self.assertEqual(self.portfolio.current_exposure_per_symbol['ES'], 45000)
        self.assertEqual(self.portfolio.current_total_exposure, 45000)
    
    def test_update_exposure_multiple_symbols(self):
        """Test exposure tracking across multiple symbols."""
        self.portfolio.update_exposure('ES', 5, 4500)   # $22,500
        self.portfolio.update_exposure('NQ', 10, 15000) # $150,000 (exceeds limit)
        
        # Second should be tracked but can_open_position should have blocked it
        self.assertEqual(self.portfolio.current_exposure_per_symbol['ES'], 22500)
        self.assertEqual(self.portfolio.current_exposure_per_symbol['NQ'], 150000)
    
    def test_update_exposure_closing_position(self):
        """Test exposure update when closing position."""
        self.portfolio.update_exposure('ES', 10, 4500)
        self.assertEqual(self.portfolio.current_total_exposure, 45000)
        
        self.portfolio.update_exposure('ES', 0, 4500)
        self.assertEqual(self.portfolio.current_exposure_per_symbol['ES'], 0.0)
        self.assertEqual(self.portfolio.current_total_exposure, 0.0)
    
    def test_update_pnl(self):
        """Test P&L tracking."""
        self.portfolio.update_pnl(realized=1000, unrealized=-500)
        
        self.assertEqual(self.portfolio.realized_pnl, 1000)
        self.assertEqual(self.portfolio.unrealized_pnl, -500)
        self.assertEqual(self.portfolio.daily_pnl, 500)
    
    def test_can_open_position_under_symbol_limit(self):
        """Test position allowed when under symbol limit."""
        result = self.portfolio.can_open_position('ES', 5, 4500)
        
        # $22,500 < $25,000 limit
        self.assertTrue(result)
    
    def test_can_open_position_exceeds_symbol_limit(self):
        """Test position blocked when exceeding symbol limit."""
        result = self.portfolio.can_open_position('ES', 10, 3000)
        
        # $30,000 > $25,000 limit
        self.assertFalse(result)
    
    def test_can_open_position_exceeds_total_limit(self):
        """Test position blocked when exceeding total limit."""
        self.portfolio.update_exposure('ES', 5, 4500)    # $22,500
        self.portfolio.update_exposure('NQ', 2, 12000)   # $24,000
        # Total: $46,500 - close to limit
        
        # Now try to add another position that would exceed limit
        result = self.portfolio.can_open_position('GC', 2, 2000)
        
        # $4,000 more would make total $50,500 - exceeds $50,000 limit
        self.assertFalse(result)
    
    def test_can_open_position_exceeds_daily_loss_limit(self):
        """Test position blocked when daily loss exceeds limit."""
        self.portfolio.update_pnl(realized=-5500, unrealized=0)
        
        result = self.portfolio.can_open_position('ES', 5, 4500)
        
        # Daily loss of -$5,500 exceeds -$5,000 limit
        self.assertFalse(result)
    
    def test_should_force_exit_false(self):
        """Test force exit not triggered when loss within limit."""
        self.portfolio.update_pnl(realized=-3000, unrealized=-1000)
        
        result = self.portfolio.should_force_exit()
        self.assertFalse(result)
    
    def test_should_force_exit_true(self):
        """Test force exit triggered when loss exceeds limit."""
        self.portfolio.update_pnl(realized=-5500, unrealized=-500)
        
        result = self.portfolio.should_force_exit()
        self.assertTrue(result)
    
    def test_reset_daily_limits(self):
        """Test daily limits reset."""
        self.portfolio.update_pnl(realized=1000, unrealized=-500)
        self.assertEqual(self.portfolio.daily_pnl, 500)
        
        self.portfolio.reset_daily_limits()
        
        self.assertEqual(self.portfolio.realized_pnl, 0.0)
        self.assertEqual(self.portfolio.unrealized_pnl, 0.0)
        self.assertEqual(self.portfolio.daily_pnl, 0.0)
    
    def test_get_available_capital(self):
        """Test available capital calculation."""
        self.portfolio.update_exposure('ES', 5, 4500)  # $22,500 used
        
        available = self.portfolio.get_available_capital()
        
        # $50,000 - $22,500 = $27,500
        self.assertEqual(available, 27500)
    
    def test_get_capital_utilization_percent(self):
        """Test capital utilization percentage."""
        self.portfolio.update_exposure('ES', 5, 4500)  # $22,500
        
        utilization = self.portfolio.get_capital_utilization_percent()
        
        # ($22,500 / $50,000) * 100 = 45%
        self.assertAlmostEqual(utilization, 45.0)
    
    def test_get_state_snapshot(self):
        """Test getting immutable state snapshot."""
        self.portfolio.update_exposure('ES', 5, 4500)
        self.portfolio.update_pnl(realized=500, unrealized=-100)
        
        snapshot = self.portfolio.get_state_snapshot()
        
        self.assertIsInstance(snapshot, PortfolioState)
        self.assertEqual(snapshot.total_capital, 100000)
        self.assertEqual(snapshot.current_total_exposure, 22500)
        self.assertEqual(snapshot.daily_pnl, 400)
        self.assertEqual(snapshot.realized_pnl, 500)
        self.assertEqual(snapshot.unrealized_pnl, -100)
    
    def test_flatten_all_positions(self):
        """Test forced flatten of all positions."""
        self.portfolio.update_exposure('ES', 5, 4500)
        self.portfolio.update_exposure('NQ', 3, 15000)
        self.assertEqual(self.portfolio.current_total_exposure, 67500)
        
        self.portfolio.flatten_all_positions()
        
        self.assertEqual(len(self.portfolio.current_exposure_per_symbol), 0)
        self.assertEqual(self.portfolio.current_total_exposure, 0.0)


class TestPortfolioRiskManagerScenarios(unittest.TestCase):
    """Test realistic trading scenarios."""
    
    def setUp(self):
        """Initialize portfolio manager."""
        self.portfolio = PortfolioRiskManager(
            total_capital=100000,
            max_symbol_exposure=25000,
            max_total_exposure=50000,
            max_daily_loss=5000
        )
    
    def test_multiple_trades_scenario(self):
        """Test realistic multi-trade scenario."""
        # Trade 1: Open ES position
        can_open_1 = self.portfolio.can_open_position('ES', 5, 4500)
        self.assertTrue(can_open_1)
        self.portfolio.update_exposure('ES', 5, 4500)
        
        # Trade 2: Open NQ position
        can_open_2 = self.portfolio.can_open_position('NQ', 2, 12000)
        self.assertTrue(can_open_2)
        self.portfolio.update_exposure('NQ', 2, 12000)
        
        # Check total exposure
        self.assertEqual(self.portfolio.current_total_exposure, 22500 + 24000)
        
        # Close ES position
        self.portfolio.update_exposure('ES', 0, 4500)
        self.assertEqual(self.portfolio.current_total_exposure, 24000)
    
    def test_progressive_loss_blocking(self):
        """Test positions blocked as losses mount."""
        # Initial position
        can_open_1 = self.portfolio.can_open_position('ES', 5, 4500)
        self.assertTrue(can_open_1)
        self.portfolio.update_exposure('ES', 5, 4500)
        
        # Loss of $2000
        self.portfolio.update_pnl(realized=-2000, unrealized=0)
        can_open_2 = self.portfolio.can_open_position('NQ', 2, 12000)
        self.assertTrue(can_open_2)  # Still within limits
        
        # Loss increased to $5500 (exceeds limit)
        self.portfolio.update_pnl(realized=-5500, unrealized=0)
        can_open_3 = self.portfolio.can_open_position('GC', 10, 2000)
        self.assertFalse(can_open_3)  # Should be blocked


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL PORTFOLIO TESTS PASSED")
        print(f"  Ran {result.testsRun} tests successfully")
    else:
        print("✗ SOME PORTFOLIO TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    print("="*70)
    
    exit(0 if result.wasSuccessful() else 1)
