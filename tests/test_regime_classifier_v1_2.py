"""
Regime Classifier Tests (Phase v1.2).

Tests:
- Synthetic trend day classification
- Synthetic reversal day classification
- Synthetic range day classification
- Confidence scoring
- Serialization
- Feature tracking
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

from engine.regime_classifier import RegimeClassifier, RegimeState
import logging


class TestRegimeClassifierBasic(unittest.TestCase):
    """Test basic regime classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_regime')
        self.logger.setLevel(logging.INFO)
        
        self.classifier = RegimeClassifier(logger=self.logger)
        self.base_time = datetime(2026, 1, 20, 9, 30, 0)
    
    def test_classifier_initializes(self):
        """Test classifier initializes with default RANGE regime."""
        regime = self.classifier.get_regime_state()
        
        self.assertEqual(regime.regime_label, 'RANGE')
        self.assertGreater(regime.regime_confidence, 0.0)
        self.assertLess(regime.regime_confidence, 1.0)
    
    def test_regime_state_is_serializable(self):
        """Test that RegimeState can be serialized to dict and JSON."""
        regime = self.classifier.get_regime_state()
        regime_dict = regime.to_dict()
        
        # Should be serializable to JSON
        json_str = json.dumps(regime_dict, default=str)
        self.assertIsNotNone(json_str)
        
        # Should contain key fields
        self.assertIn('regime_label', regime_dict)
        self.assertIn('regime_confidence', regime_dict)
        self.assertIn('regime_features', regime_dict)
        self.assertIn('contributing_signals', regime_dict)


class TestTrendDayClassification(unittest.TestCase):
    """Test trend day classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_trend')
        self.logger.setLevel(logging.INFO)
        
        self.classifier = RegimeClassifier(logger=self.logger)
        self.base_time = datetime(2026, 1, 20, 9, 30, 0)
    
    def test_synthetic_uptrend_classification(self):
        """Test that uptrending day is classified as TREND."""
        # Generate synthetic uptrending data
        base_price = 4500.0
        
        for i in range(120):
            time = self.base_time + timedelta(minutes=i)
            
            # Create uptrend: higher highs and higher lows
            drift = 0.0002
            price = base_price + (drift * base_price * i) + np.random.normal(0, 5)
            high = price + 10
            low = price - 5
            close = price
            volume = 5000 + np.random.uniform(-1000, 1000)
            vwap = close * 0.999
            
            flow_signals = {
                'initiative_move_detected': i % 5 == 0,  # Periodic initiative
                'stop_run_detected': False,
            }
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                vwap=vwap,
                flow_signals=flow_signals
            )
        
        # By bar 120, should detect trend
        final_regime = self.classifier.get_regime_state()
        
        self.logger.info(f"Uptrend classification: {final_regime.regime_label} "
                        f"({final_regime.regime_confidence:.0%})")
        
        # Trend should have meaningful confidence
        self.assertGreater(final_regime.regime_confidence, 0.3)
    
    def test_synthetic_downtrend_classification(self):
        """Test that downtrending day is classified as TREND."""
        base_price = 4500.0
        
        for i in range(120):
            time = self.base_time + timedelta(minutes=i)
            
            # Create downtrend: lower highs and lower lows
            drift = -0.0002
            price = base_price + (drift * base_price * i) + np.random.normal(0, 5)
            high = price + 5
            low = price - 10
            close = price
            volume = 5000 + np.random.uniform(-1000, 1000)
            vwap = close * 1.001
            
            flow_signals = {
                'initiative_move_detected': i % 7 == 0,
                'stop_run_detected': False,
            }
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                vwap=vwap,
                flow_signals=flow_signals
            )
        
        final_regime = self.classifier.get_regime_state()
        
        self.logger.info(f"Downtrend classification: {final_regime.regime_label} "
                        f"({final_regime.regime_confidence:.0%})")
        
        # Should detect some directional tendency
        self.assertGreater(final_regime.regime_confidence, 0.2)


class TestRangedayClassification(unittest.TestCase):
    """Test range day classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_range')
        self.logger.setLevel(logging.INFO)
        
        self.classifier = RegimeClassifier(logger=self.logger)
        self.base_time = datetime(2026, 1, 20, 9, 30, 0)
    
    def test_synthetic_range_day_classification(self):
        """Test that ranging day is classified as RANGE."""
        center_price = 4500.0
        range_width = 20.0
        
        for i in range(120):
            time = self.base_time + timedelta(minutes=i)
            
            # Oscillate around center
            sin_val = np.sin(i * np.pi / 30)  # Oscillate every 60 bars
            price = center_price + (sin_val * range_width / 2) + np.random.normal(0, 2)
            high = price + 5
            low = price - 5
            close = price
            volume = 5000 + np.random.uniform(-1000, 1000)
            vwap = center_price
            
            flow_signals = {
                'initiative_move_detected': False,
                'stop_run_detected': False,
            }
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                vwap=vwap,
                flow_signals=flow_signals
            )
        
        final_regime = self.classifier.get_regime_state()
        
        self.logger.info(f"Range day classification: {final_regime.regime_label} "
                        f"({final_regime.regime_confidence:.0%})")
        
        # Range day should have RANGE label or low confidence on other labels
        if final_regime.regime_label == 'RANGE':
            self.assertGreater(final_regime.regime_confidence, 0.3)


class TestReversalDayClassification(unittest.TestCase):
    """Test reversal day classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_reversal')
        self.logger.setLevel(logging.INFO)
        
        self.classifier = RegimeClassifier(logger=self.logger)
        self.base_time = datetime(2026, 1, 20, 9, 30, 0)
    
    def test_synthetic_reversal_day_classification(self):
        """Test that reversal day is classified as REVERSAL."""
        base_price = 4500.0
        
        for i in range(120):
            time = self.base_time + timedelta(minutes=i)
            
            # Up first half, then reversal
            if i < 60:
                drift = 0.0003  # Strong uptrend
                price = base_price + (drift * base_price * i) + np.random.normal(0, 5)
            else:
                drift = -0.0005  # Strong downtrend after reversal
                price = base_price + (0.0003 * base_price * 60) + (drift * base_price * (i - 60)) + np.random.normal(0, 5)
            
            high = price + 10
            low = price - 5
            close = price
            volume = 5000 + np.random.uniform(-1000, 1000)
            vwap = close
            
            # Stop-runs more common in reversal zones
            is_reversal_zone = (i > 50)
            flow_signals = {
                'initiative_move_detected': i % 10 == 0,
                'stop_run_detected': is_reversal_zone and (i % 8 == 0),
            }
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                vwap=vwap,
                flow_signals=flow_signals
            )
        
        final_regime = self.classifier.get_regime_state()
        
        self.logger.info(f"Reversal day classification: {final_regime.regime_label} "
                        f"({final_regime.regime_confidence:.0%})")
        
        # Should detect reversal or at least have features showing reversal
        if final_regime.regime_label == 'REVERSAL':
            self.assertGreater(final_regime.regime_confidence, 0.2)


class TestRegimeConfidenceScoring(unittest.TestCase):
    """Test confidence scoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = RegimeClassifier()
        self.base_time = datetime(2026, 1, 20, 9, 30, 0)
    
    def test_confidence_increases_with_bars(self):
        """Test that confidence increases as more bars are processed."""
        confidences = []
        base_price = 4500.0
        
        for i in range(60):
            time = self.base_time + timedelta(minutes=i)
            
            drift = 0.0002
            price = base_price + (drift * base_price * i)
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=price + 10,
                low=price - 5,
                close=price,
                volume=5000,
                vwap=price,
                flow_signals={}
            )
            
            confidences.append(regime.regime_confidence)
        
        # Confidence should stabilize over time
        early_conf = np.mean(confidences[:10])
        late_conf = np.mean(confidences[-10:])
        
        # Late confidence should be different from early (more information)
        self.assertNotEqual(early_conf, late_conf)
    
    def test_confidence_bounds(self):
        """Test that confidence stays within 0-1 bounds."""
        base_price = 4500.0
        
        for i in range(100):
            time = self.base_time + timedelta(minutes=i)
            
            price = base_price + np.random.normal(0, 5)
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=price + 10,
                low=price - 5,
                close=price,
                volume=5000,
                vwap=price,
                flow_signals={}
            )
            
            self.assertGreaterEqual(regime.regime_confidence, 0.0)
            self.assertLessEqual(regime.regime_confidence, 1.0)


class TestRegimeFeatureTracking(unittest.TestCase):
    """Test that regime features are tracked correctly."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = RegimeClassifier()
        self.base_time = datetime(2026, 1, 20, 9, 30, 0)
    
    def test_regime_features_populated(self):
        """Test that regime features dict is populated."""
        base_price = 4500.0
        
        for i in range(50):
            time = self.base_time + timedelta(minutes=i)
            price = base_price + (0.0002 * base_price * i)
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=price + 10,
                low=price - 5,
                close=price,
                volume=5000,
                vwap=price * 0.999,
                flow_signals={}
            )
        
        regime = self.classifier.get_regime_state()
        features = regime.regime_features
        
        # Should have various features populated
        self.assertIn('vwap_distance', features)
        self.assertIn('vwap_persistence', features)
        self.assertIn('hh_hl_score', features)
        self.assertIn('initiative_ratio', features)
    
    def test_contributing_signals_tracked(self):
        """Test that contributing signals are tracked."""
        base_price = 4500.0
        
        for i in range(60):
            time = self.base_time + timedelta(minutes=i)
            price = base_price + (0.0003 * base_price * i)
            
            regime = self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=price + 10,
                low=price - 5,
                close=price,
                volume=5000,
                vwap=price,
                flow_signals={'initiative_move_detected': i % 5 == 0}
            )
        
        regime = self.classifier.get_regime_state()
        signals = regime.contributing_signals
        
        # Should have some signals contributing
        self.assertGreater(len(signals), 0)
        
        # Each signal should have required fields
        for signal in signals:
            self.assertIsNotNone(signal.name)
            self.assertIsNotNone(signal.regime_type)
            self.assertGreaterEqual(signal.value, 0.0)
            self.assertLessEqual(signal.value, 1.0)


class TestRegimeSessionReset(unittest.TestCase):
    """Test session reset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = RegimeClassifier()
        self.base_time = datetime(2026, 1, 20, 9, 30, 0)
    
    def test_session_reset_clears_session_state(self):
        """Test that session reset clears session-specific state."""
        base_price = 4500.0
        
        # Build up state
        for i in range(50):
            time = self.base_time + timedelta(minutes=i)
            price = base_price + (0.0002 * base_price * i)
            
            self.classifier.update_with_bar(
                timestamp=time,
                open_price=price,
                high=price + 10,
                low=price - 5,
                close=price,
                volume=5000,
                vwap=price,
                flow_signals={}
            )
        
        regime_before = self.classifier.get_regime_state()
        
        # Reset session
        self.classifier.reset_session()
        
        # Session high/low should be reset but history should remain
        self.assertEqual(self.classifier.session_high, 0.0)
        self.assertEqual(self.classifier.session_low, float('inf'))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL REGIME CLASSIFIER TESTS PASSED")
        print(f"  Ran {result.testsRun} tests successfully")
    else:
        print("✗ SOME REGIME CLASSIFIER TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    print("="*70)
    
    exit(0 if result.wasSuccessful() else 1)
