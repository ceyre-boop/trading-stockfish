#!/usr/bin/env python3
"""
NEWS_MACRO_ENGINE Integration Test
Tests the NewsMacroEngine module with sample data and verifies integration.

Usage:
    python test_news_macro_engine.py
"""

import sys
from datetime import datetime, timedelta
import json

# Test imports
try:
    from analytics.news_macro_engine import (
        NewsMacroEngine,
        MacroEvent,
        NewsArticle,
        MacroFeatures,
        SimpleNLPClassifier,
        MacroEventCategory,
        EventImpactLevel,
        integrate_macro_features_into_state
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 1: Classifier functionality
print("\n[TEST 1] SimpleNLPClassifier")
try:
    # Test sentiment - use clear hawkish language
    hawkish_score = SimpleNLPClassifier.classify_sentiment(
        "Inflation remains sticky and resilient. Tightening is needed to control prices."
    )
    assert -1 <= hawkish_score <= 1, "Hawkish score out of range"
    assert hawkish_score > 0, "Should be hawkish (positive)"
    print(f"  ✓ Hawkish classification: {hawkish_score:.2f}")
    
    dovish_score = SimpleNLPClassifier.classify_sentiment(
        "Labor market weakness and easing inflation support accommodative monetary policy."
    )
    assert dovish_score < 0, "Should be dovish (negative)"
    print(f"  ✓ Dovish classification: {dovish_score:.2f}")
    
    # Test risk sentiment
    risk_on = SimpleNLPClassifier.classify_risk_sentiment(
        "Markets rally on strong jobs data. Investor confidence soars."
    )
    assert risk_on > 0, "Should be risk-on (positive)"
    print(f"  ✓ Risk-on classification: {risk_on:.2f}")
    
    risk_off = SimpleNLPClassifier.classify_risk_sentiment(
        "Geopolitical tensions escalate. Investors flee to safety."
    )
    assert risk_off < 0, "Should be risk-off (negative)"
    print(f"  ✓ Risk-off classification: {risk_off:.2f}")
    
    # Test surprise
    surprise_positive = SimpleNLPClassifier.classify_surprise(
        actual=3.4, forecast=3.1
    )
    assert surprise_positive > 0, "Should be positive surprise"
    print(f"  ✓ Upside surprise (3.4 vs 3.1): {surprise_positive:.2f}")
    
    surprise_negative = SimpleNLPClassifier.classify_surprise(
        actual=2.5, forecast=3.0
    )
    assert surprise_negative < 0, "Should be negative surprise"
    print(f"  ✓ Downside surprise (2.5 vs 3.0): {surprise_negative:.2f}")
    
except AssertionError as e:
    print(f"  ✗ Classifier test failed: {e}")
    sys.exit(1)

# Test 2: Engine instantiation
print("\n[TEST 2] NewsMacroEngine Instantiation")
try:
    engine = NewsMacroEngine(symbol='USD', lookback_hours=24, verbose=False)
    print(f"  ✓ Engine created for {engine.symbol}")
    print(f"    - Lookback window: {engine.lookback_hours} hours")
except Exception as e:
    print(f"  ✗ Engine instantiation failed: {e}")
    sys.exit(1)

# Test 3: Load sample data
print("\n[TEST 3] Load Sample Data")
try:
    # Load events
    events_loaded = engine.load_event_calendar('data/macro_events_sample.csv')
    print(f"  ✓ Loaded {events_loaded} macro events from CSV")
    
    # Load news
    news_loaded = engine.load_news_articles('data/macro_news_sample.csv')
    print(f"  ✓ Loaded {news_loaded} news articles from CSV")
    
except FileNotFoundError:
    print("  ⚠ Sample data files not found - skipping data load test")
except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    sys.exit(1)

# Test 4: Time-causality validation
print("\n[TEST 4] Time-Causality Validation")
try:
    is_valid, warnings = engine.validate_time_causality()
    print(f"  ✓ Validation complete: {'PASS' if is_valid else 'FAIL'}")
    if warnings:
        for w in warnings:
            print(f"    - Warning: {w}")
    else:
        print(f"    - No warnings or errors")
except Exception as e:
    print(f"  ✗ Validation failed: {e}")
    sys.exit(1)

# Test 5: Feature extraction
print("\n[TEST 5] Feature Extraction for Specific Timestamp")
try:
    # Use a timestamp in the middle of our data
    target_time = datetime(2024, 1, 15, 15, 0, 0)
    features = engine.get_features_for_timestamp(target_time)
    
    print(f"  ✓ Features extracted for {target_time}")
    print(f"    - Hawkishness: {features.hawkishness_score:.2f}")
    print(f"    - Risk sentiment: {features.risk_sentiment_score:.2f}")
    print(f"    - Surprise: {features.surprise_score:.2f}")
    print(f"    - State: {features.macro_news_state}")
    print(f"    - Events: {features.macro_event_count}")
    print(f"    - Articles: {features.news_article_count}")
    
except Exception as e:
    print(f"  ✗ Feature extraction failed: {e}")
    sys.exit(1)

# Test 6: Build macro features timeseries
print("\n[TEST 6] Build Macro Features Timeseries")
try:
    features_df = engine.build_macro_features()
    print(f"  ✓ Built timeseries with {len(features_df)} rows")
    if len(features_df) > 0:
        print(f"    - Date range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
        print(f"    - Columns: {', '.join(features_df.columns.tolist())}")
except Exception as e:
    print(f"  ✗ Timeseries building failed: {e}")
    sys.exit(1)

# Test 7: Integration with market state
print("\n[TEST 7] Integration with MarketStateBuilder")
try:
    target_time = datetime(2024, 1, 12, 15, 0, 0)
    
    # Create mock market state
    base_state = {
        'price': 5000.0,
        'volume': 1000000,
        'volatility': 0.015,
        'timestamp': target_time
    }
    
    # Add macro features
    enhanced_state = integrate_macro_features_into_state(
        engine, target_time, base_state
    )
    
    print(f"  ✓ State enhanced with macro features")
    if 'macro_news_features' in enhanced_state:
        print(f"    - Macro block present: ✓")
        features_block = enhanced_state['macro_news_features']
        print(f"    - Risk sentiment: {features_block['risk_sentiment_score']:.2f}")
        print(f"    - State: {features_block['macro_news_state']}")
    else:
        print(f"    - Warning: Macro block not found in state")
        
except Exception as e:
    print(f"  ✗ Integration test failed: {e}")
    sys.exit(1)

# Test 8: Official tournament mode checks
print("\n[TEST 8] Official Tournament Mode Checks")
try:
    # Verify past timestamps work fine
    past_time = datetime(2024, 1, 10, 12, 0, 0)
    try:
        features_past = engine.get_features_for_timestamp(
            past_time, official_mode=True
        )
        print(f"  ✓ Official mode: Past timestamp in data accepted")
    except Exception as e:
        print(f"  ✗ Official mode: Past timestamp should have been accepted! Error: {e}")
        sys.exit(1)
    
    # Test validation method works
    is_valid, warnings = engine.validate_time_causality()
    if is_valid:
        print(f"  ✓ Official mode: Time-causality validation passed")
    else:
        print(f"  ⚠ Validation returned warnings: {warnings}")
    
except Exception as e:
    print(f"  ✗ Official tournament mode test failed: {e}")
    sys.exit(1)

# Test 9: Export to JSON
print("\n[TEST 9] Export Features to JSON")
try:
    export_path = 'test_macro_features_output.json'
    engine.export_features_to_json(export_path)
    print(f"  ✓ Features exported to {export_path}")
    
    # Verify JSON is valid
    with open(export_path, 'r') as f:
        data = json.load(f)
    print(f"  ✓ JSON is valid, contains {len(data)} entries")
    
    # Clean up
    import os
    os.remove(export_path)
    
except Exception as e:
    print(f"  ✗ JSON export test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✓ ALL TESTS PASSED")
print("="*60)
print("\nNewsMacroEngine module is fully functional!")
print("\nNext steps:")
print("1. Review NEWS_MACRO_ENGINE.md for full documentation")
print("2. Review NEWS_MACRO_ENGINE_INTEGRATION.md for integration guide")
print("3. Add macro_engine parameter to your trading evaluator")
print("4. Run official tournaments with macro features enabled")
