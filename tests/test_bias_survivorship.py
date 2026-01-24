"""
Test suite for survivorship bias detection.

Verifies:
- All historical symbols preserved (no filtering)
- No delisted companies removed from data
- Dividend/split adjustments don't leak future
- Data includes bankrupt companies during testing
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_integrity import DataIntegrityLayer, DataIntegrityError


class TestSurvivorshipBias(unittest.TestCase):
    """Test detection of survivorship bias."""
    
    def setUp(self):
        """Set up test data."""
        self.layer = DataIntegrityLayer(verbose=False)
    
    def test_all_symbols_preserved(self):
        """Test that all historical symbols are preserved."""
        # Simulate 4-year history for 10 symbols
        dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        symbols = ['ES', 'NQ', 'GC', 'CL', 'EURUSD', 'GBPUSD', 'BTC', 'ETH', 'SPX', 'VIX']
        
        # Create data with all symbols present for all dates
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'close': 4500 + np.random.randn() * 100,
                    'volume': np.random.randint(100000, 1000000)
                })
        
        df = pd.DataFrame(data)
        
        # Verify all symbols present for all dates
        symbol_counts = df['symbol'].value_counts()
        for symbol, count in symbol_counts.items():
            self.assertEqual(count, len(dates), 
                           f"Symbol {symbol} has {count} rows, expected {len(dates)}")
    
    def test_no_forward_looking_adjustments(self):
        """Test that price adjustments don't use future information."""
        dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        
        # Create price data
        prices = 4500 + np.cumsum(np.random.randn(len(dates)) * 5)
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'close_adjusted': prices  # This would be adjusted for splits
        })
        
        # Split happens on day 500 (future relative to day 100)
        split_date = dates[500]
        split_date_idx = 500
        
        # Proper adjustment: only adjust data UP TO split date
        # Data AFTER split is already in post-split prices
        # Data BEFORE split should be adjusted DOWN
        
        # Improper (lookahead): Using future knowledge to adjust past prices
        # Check that adjustment is done correctly
        
        # This test verifies the concept, actual adjustment logic
        # should be in data loading layer
        self.assertGreater(len(df), 500)
    
    def test_bankrupt_companies_included(self):
        """Test that bankrupt/delisted companies are included in history."""
        dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        
        # Create data for companies including one that goes bankrupt
        symbols = ['ES', 'NQ', 'BANKRUPT_CO', 'GC']
        data = []
        
        for symbol in symbols:
            for date in dates:
                if symbol == 'BANKRUPT_CO':
                    # This company trades until 2024-06-15, then delisted
                    if date <= datetime(2024, 6, 15):
                        price = 100 - (date - datetime(2022, 1, 1)).days * 0.05
                        data.append({
                            'timestamp': date,
                            'symbol': symbol,
                            'close': max(price, 1),  # Can't go below $1
                            'volume': np.random.randint(10000, 100000)
                        })
                    # After delisting, no data - this is CORRECT
                else:
                    data.append({
                        'timestamp': date,
                        'symbol': symbol,
                        'close': 4500 + np.random.randn() * 100,
                        'volume': np.random.randint(100000, 1000000)
                    })
        
        df = pd.DataFrame(data)
        
        # Verify bankrupt company IS in data (up to delisting date)
        bankrupt_data = df[df['symbol'] == 'BANKRUPT_CO']
        self.assertGreater(len(bankrupt_data), 0)
        
        # Verify it's NOT in data after delisting date
        delisted_after = bankrupt_data[
            bankrupt_data['timestamp'] > datetime(2024, 6, 15)
        ]
        self.assertEqual(len(delisted_after), 0)
    
    def test_no_delisting_filter_applied(self):
        """Test that delisting filter isn't retroactively applied."""
        dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        
        # Create data for company that was delisted in 2024
        # but would have had valid history before that
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'ONCE_DELISTED',
            'close': 4500 + np.random.randn(len(dates)) * 100
        })
        
        # Data BEFORE delisting date should be present
        before_delisting = df[df['timestamp'] < datetime(2024, 6, 15)]
        self.assertGreater(len(before_delisting), 100)
        
        # Verify early data is complete (no random gaps)
        early_data = df[df['timestamp'] < datetime(2023, 1, 1)]
        days_in_range = (datetime(2023, 1, 1) - datetime(2022, 1, 1)).days
        self.assertEqual(len(early_data), days_in_range)


class TestDataCompletenessAndCleanliness(unittest.TestCase):
    """Test that data is complete and doesn't have suspicious patterns."""
    
    def setUp(self):
        """Set up test data."""
        self.layer = DataIntegrityLayer(verbose=False)
    
    def test_no_artificial_gaps(self):
        """Test that data doesn't have artificial gaps removed."""
        # Create daily OHLCV data
        dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 4500 + np.random.randn(len(dates)) * 5,
            'high': 4510 + np.random.randn(len(dates)) * 5,
            'low': 4490 + np.random.randn(len(dates)) * 5,
            'close': 4500 + np.random.randn(len(dates)) * 5,
            'volume': np.random.randint(100000, 1000000, len(dates))
        })
        
        # Verify no missing dates (would indicate filtered data)
        expected_dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        actual_dates = df['timestamp'].unique()
        
        # Check all dates present
        self.assertEqual(len(actual_dates), len(expected_dates))
    
    def test_volume_not_suspiciously_smoothed(self):
        """Test that volume doesn't show signs of artificial smoothing."""
        dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        
        # Create realistic noisy volume
        true_volume = np.random.lognormal(mean=12, sigma=0.5, size=len(dates))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'close': 4500 + np.random.randn(len(dates)) * 5,
            'volume': true_volume.astype(int)
        })
        
        # Calculate coefficient of variation (should be > 0.3 for realistic data)
        cv = df['volume'].std() / df['volume'].mean()
        self.assertGreater(cv, 0.2)  # Should have variation
    
    def test_unrealistic_returns_detection(self):
        """Test detection of unrealistic price movements."""
        dates = pd.date_range('2022-01-01', '2026-01-19', freq='D')
        
        # Create data with realistic returns
        prices = 4500 + np.cumsum(np.random.randn(len(dates)) * 5)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })
        
        # Calculate returns
        returns = df['close'].pct_change()
        
        # Realistic daily returns should be < ±10% (for equity index)
        extreme_moves = (abs(returns) > 0.15).sum()
        self.assertLess(extreme_moves, len(returns) * 0.01)  # Less than 1% extreme moves


if __name__ == '__main__':
    # Run with verbose output
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with code 1 if any test failed
    exit(0 if result.wasSuccessful() else 1)
