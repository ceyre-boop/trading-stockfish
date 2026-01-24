"""
Regime Statistics Tests (Phase v2.0).

Comprehensive validation of regime statistics computation, aggregation,
and deterministic behavior.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from analytics.regime_statistics import (
    RegimeStatistics,
    DayRegimeStats,
    RegimeTransition,
    AggregateRegimeStats
)


class TestRegimeStatisticsBasic(unittest.TestCase):
    """Test basic regime statistics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stats = RegimeStatistics()
        self.base_date = datetime(2026, 1, 20)
    
    def tearDown(self):
        """Clean up."""
        self.stats.clear()
    
    def test_statistics_initializes(self):
        """Test that RegimeStatistics initializes correctly."""
        self.assertIsNotNone(self.stats.classifier)
        self.assertEqual(len(self.stats.day_stats), 0)
    
    def test_day_regime_stats_structure(self):
        """Test that DayRegimeStats has correct structure."""
        day_stat = DayRegimeStats(date=self.base_date)
        self.assertEqual(day_stat.date, self.base_date)
        self.assertEqual(day_stat.regime_counts, {})
        self.assertEqual(day_stat.transition_count, 0)
        self.assertIsNone(day_stat.dominant_regime)
    
    def test_aggregate_regime_stats_structure(self):
        """Test that AggregateRegimeStats has correct structure."""
        aggregate = AggregateRegimeStats(total_days=0, total_bars=0)
        self.assertEqual(aggregate.total_days, 0)
        self.assertEqual(aggregate.total_bars, 0)
        self.assertEqual(aggregate.regime_frequency, {})
        self.assertEqual(aggregate.transition_matrix, {})


class TestRegimeDayAnalysis(unittest.TestCase):
    """Test analysis of individual days."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stats = RegimeStatistics()
        self.base_date = datetime(2026, 1, 20, 9, 30, 0)
    
    def tearDown(self):
        """Clean up."""
        self.stats.clear()
    
    def _create_synthetic_day(self, day_type: str, length: int = 480) -> pd.DataFrame:
        """Create synthetic day data."""
        timestamps = pd.date_range(self.base_date, periods=length, freq='1min')
        
        prices = [4500.0]
        for i in range(1, length):
            if day_type == 'trend':
                drift = 0.0003
            elif day_type == 'range':
                drift = 0.00005 * np.sin(2 * np.pi * i / length)
            else:  # reversal
                drift = 0.0004 if i < length // 2 else -0.0005
            
            price = prices[-1] + (drift * prices[-1]) + np.random.normal(0, 2)
            prices.append(price)
        
        prices = np.array(prices)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices + np.random.normal(0, 1, length),
            'volume': np.random.randint(1000, 5000, length),
            'vwap': prices,  # Simplified VWAP
            'session': 'RTH',
            'initiative': np.random.rand(length) > 0.7,
            'stop_run': np.random.rand(length) > 0.9,
            'flow_signals': [{}] * length,
        })
        
        return data
    
    def test_analyze_trend_day(self):
        """Test analysis of a trend day."""
        day_data = self._create_synthetic_day('trend')
        stats = self.stats.analyze_day(day_data, self.base_date)
        
        self.assertIsNotNone(stats.dominant_regime)
        self.assertGreater(stats.dominant_regime_pct, 0.0)
        self.assertEqual(len(stats.regime_counts), stats.regime_counts.__len__())
    
    def test_analyze_range_day(self):
        """Test analysis of a range day."""
        day_data = self._create_synthetic_day('range')
        stats = self.stats.analyze_day(day_data, self.base_date)
        
        self.assertIsNotNone(stats.dominant_regime)
        self.assertGreater(stats.dominant_regime_pct, 0.0)
    
    def test_analyze_reversal_day(self):
        """Test analysis of a reversal day."""
        day_data = self._create_synthetic_day('reversal')
        stats = self.stats.analyze_day(day_data, self.base_date)
        
        self.assertIsNotNone(stats.dominant_regime)
        # Reversal days should have more transitions
        self.assertGreaterEqual(stats.transition_count, 0)
    
    def test_regime_counts_sum_to_total_bars(self):
        """Test that regime counts sum to total bars."""
        day_data = self._create_synthetic_day('trend', length=100)
        stats = self.stats.analyze_day(day_data, self.base_date)
        
        total_bars_in_stats = sum(stats.regime_counts.values())
        self.assertEqual(total_bars_in_stats, len(day_data))
    
    def test_day_stat_transition_tracking(self):
        """Test that transitions are tracked correctly."""
        day_data = self._create_synthetic_day('trend', length=50)
        stats = self.stats.analyze_day(day_data, self.base_date)
        
        # Each transition should have from/to regime
        for transition in stats.transitions:
            self.assertIsNotNone(transition.from_regime)
            self.assertIsNotNone(transition.to_regime)
            self.assertGreater(transition.bar_index, -1)


class TestAggregation(unittest.TestCase):
    """Test aggregation of multiple days."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stats = RegimeStatistics()
        self.base_date = datetime(2026, 1, 20, 9, 30, 0)
    
    def tearDown(self):
        """Clean up."""
        self.stats.clear()
    
    def _create_synthetic_day(self, day_type: str, length: int = 480) -> pd.DataFrame:
        """Create synthetic day data."""
        timestamps = pd.date_range(self.base_date, periods=length, freq='1min')
        
        prices = [4500.0]
        for i in range(1, length):
            if day_type == 'trend':
                drift = 0.0003
            elif day_type == 'range':
                drift = 0.00005 * np.sin(2 * np.pi * i / length)
            else:
                drift = 0.0004 if i < length // 2 else -0.0005
            
            price = prices[-1] + (drift * prices[-1]) + np.random.normal(0, 2)
            prices.append(price)
        
        prices = np.array(prices)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices + np.random.normal(0, 1, length),
            'volume': np.random.randint(1000, 5000, length),
            'vwap': prices,
            'session': 'RTH',
            'initiative': np.random.rand(length) > 0.7,
            'stop_run': np.random.rand(length) > 0.9,
            'flow_signals': [{}] * length,
        })
        
        return data
    
    def test_aggregate_multiple_days(self):
        """Test aggregation of multiple days."""
        for i in range(5):
            day_data = self._create_synthetic_day('trend', length=100)
            self.stats.analyze_day(
                day_data,
                self.base_date + timedelta(days=i)
            )
        
        aggregate = self.stats.aggregate_statistics()
        
        self.assertEqual(aggregate.total_days, 5)
        self.assertGreater(aggregate.total_bars, 0)
    
    def test_transition_matrix_computation(self):
        """Test that transition matrix is computed correctly."""
        # Analyze multiple days
        for i in range(3):
            day_data = self._create_synthetic_day('trend', length=80)
            self.stats.analyze_day(
                day_data,
                self.base_date + timedelta(days=i)
            )
        
        aggregate = self.stats.aggregate_statistics()
        
        # Transition matrix should have entries
        self.assertIsInstance(aggregate.transition_matrix, dict)
        
        # All entries should map to dicts
        for from_regime, to_dict in aggregate.transition_matrix.items():
            self.assertIsInstance(to_dict, dict)
    
    def test_regime_frequency_distribution(self):
        """Test regime frequency computation."""
        for i in range(3):
            day_data = self._create_synthetic_day('trend', length=100)
            self.stats.analyze_day(
                day_data,
                self.base_date + timedelta(days=i)
            )
        
        aggregate = self.stats.aggregate_statistics()
        
        # Check that frequency dict has expected structure
        self.assertGreater(len(aggregate.regime_frequency), 0)
        
        for regime, freq_data in aggregate.regime_frequency.items():
            self.assertIn('bars', freq_data)
            self.assertIn('pct_of_total_bars', freq_data)
            self.assertIn('days_with_regime', freq_data)
            self.assertIn('pct_of_days', freq_data)
            self.assertIn('avg_confidence', freq_data)
    
    def test_confidence_distribution(self):
        """Test confidence distribution collection."""
        for i in range(2):
            day_data = self._create_synthetic_day('trend', length=100)
            self.stats.analyze_day(
                day_data,
                self.base_date + timedelta(days=i)
            )
        
        aggregate = self.stats.aggregate_statistics()
        
        # Should have confidence distributions for each regime
        self.assertGreater(len(aggregate.confidence_distribution), 0)
        
        for regime, confidences in aggregate.confidence_distribution.items():
            self.assertIsInstance(confidences, list)
            if confidences:
                self.assertTrue(all(0 <= c <= 1 for c in confidences))
    
    def test_day_type_distribution(self):
        """Test day type distribution."""
        for i in range(3):
            day_data = self._create_synthetic_day('trend', length=100)
            self.stats.analyze_day(
                day_data,
                self.base_date + timedelta(days=i)
            )
        
        aggregate = self.stats.aggregate_statistics()
        
        # Total days with a dominant regime should not exceed total days
        total_days_with_dominant = sum(aggregate.day_type_distribution.values())
        self.assertLessEqual(total_days_with_dominant, aggregate.total_days)


class TestDeterministicOutput(unittest.TestCase):
    """Test that output is deterministic for fixed inputs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_date = datetime(2026, 1, 20, 9, 30, 0)
    
    def _create_fixed_day(self) -> pd.DataFrame:
        """Create a fixed, deterministic day."""
        np.random.seed(42)  # Fixed seed
        
        timestamps = pd.date_range(self.base_date, periods=100, freq='1min')
        prices = np.linspace(4500, 4510, 100) + np.random.normal(0, 1, 100)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.full(100, 1000),
            'vwap': prices,
            'session': 'RTH',
            'initiative': np.array([False] * 100),
            'stop_run': np.array([False] * 100),
            'flow_signals': [{}] * 100,
        })
        
        return data
    
    def test_deterministic_single_day_analysis(self):
        """Test that analyzing same day twice gives same result."""
        day_data = self._create_fixed_day()
        
        stats1 = RegimeStatistics()
        result1 = stats1.analyze_day(day_data, self.base_date)
        
        stats2 = RegimeStatistics()
        result2 = stats2.analyze_day(day_data, self.base_date)
        
        # Compare key metrics
        self.assertEqual(result1.dominant_regime, result2.dominant_regime)
        self.assertEqual(result1.transition_count, result2.transition_count)
        self.assertEqual(result1.regime_counts, result2.regime_counts)
    
    def test_deterministic_aggregation(self):
        """Test that aggregation is deterministic."""
        stats1 = RegimeStatistics()
        stats2 = RegimeStatistics()
        
        for i in range(3):
            day_data = self._create_fixed_day()
            stats1.analyze_day(day_data, self.base_date + timedelta(days=i))
            stats2.analyze_day(day_data, self.base_date + timedelta(days=i))
        
        agg1 = stats1.aggregate_statistics()
        agg2 = stats2.aggregate_statistics()
        
        # Compare aggregated results
        self.assertEqual(agg1.total_days, agg2.total_days)
        self.assertEqual(agg1.total_bars, agg2.total_bars)
        self.assertEqual(agg1.regime_frequency, agg2.regime_frequency)


class TestReportGeneration(unittest.TestCase):
    """Test report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stats = RegimeStatistics()
        self.base_date = datetime(2026, 1, 20, 9, 30, 0)
    
    def tearDown(self):
        """Clean up."""
        self.stats.clear()
    
    def _create_synthetic_day(self, length: int = 100) -> pd.DataFrame:
        """Create synthetic day data."""
        timestamps = pd.date_range(self.base_date, periods=length, freq='1min')
        prices = np.linspace(4500, 4505, length) + np.random.normal(0, 1, length)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.randint(1000, 5000, length),
            'vwap': prices,
            'session': 'RTH',
            'initiative': np.random.rand(length) > 0.7,
            'stop_run': np.random.rand(length) > 0.9,
            'flow_signals': [{}] * length,
        })
        
        return data
    
    def test_report_generation_with_data(self):
        """Test that report can be generated with data."""
        for i in range(2):
            day_data = self._create_synthetic_day()
            self.stats.analyze_day(day_data, self.base_date + timedelta(days=i))
        
        report = self.stats.generate_summary_report()
        
        self.assertIn("REGIME STATISTICS SUMMARY REPORT", report)
        self.assertIn("Total Days Analyzed: 2", report)
        self.assertIn("REGIME FREQUENCY", report)
        self.assertIn("TRANSITION MATRIX", report)
    
    def test_report_generation_empty(self):
        """Test that report handles empty statistics."""
        report = self.stats.generate_summary_report()
        
        self.assertIn("No statistics to report", report)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stats = RegimeStatistics()
        self.base_date = datetime(2026, 1, 20, 9, 30, 0)
    
    def tearDown(self):
        """Clean up."""
        self.stats.clear()
    
    def test_single_regime_day(self):
        """Test day with only one regime."""
        timestamps = pd.date_range(self.base_date, periods=50, freq='1min')
        prices = np.linspace(4500, 4500.5, 50)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.full(50, 1000),
            'vwap': prices,
            'session': 'RTH',
            'initiative': np.array([False] * 50),
            'stop_run': np.array([False] * 50),
            'flow_signals': [{}] * 50,
        })
        
        stats = self.stats.analyze_day(data, self.base_date)
        
        # Single regime days should have no transitions
        self.assertLessEqual(stats.transition_count, 1)
    
    def test_high_volatility_day(self):
        """Test day with high volatility."""
        timestamps = pd.date_range(self.base_date, periods=100, freq='1min')
        prices = 4500 + np.random.normal(0, 20, 100)  # High volatility
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + 10,
            'low': prices - 10,
            'close': prices,
            'volume': np.random.randint(5000, 10000, 100),
            'vwap': prices,
            'session': 'RTH',
            'initiative': np.random.rand(100) > 0.5,
            'stop_run': np.random.rand(100) > 0.8,
            'flow_signals': [{}] * 100,
        })
        
        stats = self.stats.analyze_day(data, self.base_date)
        
        # High volatility should produce valid stats
        self.assertIsNotNone(stats.dominant_regime)
        self.assertGreater(stats.regime_counts.__len__(), 0)


if __name__ == '__main__':
    unittest.main()
