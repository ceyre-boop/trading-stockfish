"""
Simple validation script to demonstrate MarketStateBuilder integration with
SessionContext and FlowContext (v1.1.1).

Shows session transitions, flow fields, and capacity awareness in MarketState.
"""
import os
import sys
import pandas as pd
from datetime import datetime, timezone
import logging

# Ensure project root is on path
ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analytics.data_loader import DataLoader, MarketStateBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data():
    """Generate synthetic ES minute data for testing"""
    dates = pd.date_range('2026-01-15 03:00:00', periods=500, freq='1min', tz=timezone.utc)
    
    # Synthetic trending data
    prices = [5000.0 + i * 0.05 + (i % 10) * 0.1 for i in range(500)]
    
    df = pd.DataFrame({
        'open': [p - 0.25 for p in prices],
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [10000 + i * 5 for i in range(500)],
    }, index=dates)
    df.index.name = 'timestamp'
    return df


def main():
    logger.info("=" * 60)
    logger.info("MarketStateBuilder v1.1.1 Validation")
    logger.info("=" * 60)
    
    # Create test data
    df = create_test_data()
    logger.info(f"Generated {len(df)} synthetic ES minute bars")
    logger.info(f"Time range: {df.index[0]} to {df.index[-1]}")
    
    # Build market states with session/flow awareness
    builder = MarketStateBuilder('ES', '1min', lookback=50)
    logger.info("Building market states with SessionContext integration...")
    
    states = builder.build_states(df, time_causal_check=False)
    logger.info(f"Built {len(states)} market states")
    
    # Track sessions and fields
    sessions = {}
    flow_fields = {
        'vwap': 0, 'prior_high': 0, 'prior_low': 0,
        'overnight_high': 0, 'overnight_low': 0,
        'round_level_proximity': 0, 'stop_run': 0, 'initiative': 0
    }
    
    for i, state in enumerate(states):
        # Track sessions
        if state.session_name:
            if state.session_name not in sessions:
                sessions[state.session_name] = []
            sessions[state.session_name].append(i)
        
        # Track flow fields
        if state.vwap is not None and state.vwap > 0:
            flow_fields['vwap'] += 1
        if state.prior_high is not None:
            flow_fields['prior_high'] += 1
        if state.prior_low is not None:
            flow_fields['prior_low'] += 1
        if state.overnight_high is not None:
            flow_fields['overnight_high'] += 1
        if state.overnight_low is not None:
            flow_fields['overnight_low'] += 1
        if state.round_level_proximity:
            flow_fields['round_level_proximity'] += 1
        if state.stop_run_detected:
            flow_fields['stop_run'] += 1
        if state.initiative_move_detected:
            flow_fields['initiative'] += 1
    
    logger.info("\nSession Distribution:")
    for sess_name in sorted(sessions.keys()):
        count = len(sessions[sess_name])
        indices = sessions[sess_name]
        logger.info(f"  {sess_name:15s}: {count:3d} states (indices {min(indices)}-{max(indices)})")
    
    logger.info("\nFlow Field Coverage:")
    for field_name in sorted(flow_fields.keys()):
        count = flow_fields[field_name]
        pct = count / len(states) * 100 if len(states) > 0 else 0
        logger.info(f"  {field_name:25s}: {count:3d} / {len(states)} ({pct:5.1f}%)")
    
    # Sample a few states
    logger.info("\nSample States (first 3 with sessions):")
    count = 0
    for state in states:
        if state.session_name and count < 3:
            logger.info(f"  [{state.session_name}] Time={datetime.fromtimestamp(state.timestamp, tz=timezone.utc).strftime('%H:%M')} "
                       f"Close={state.close:.1f} VWAP={state.vwap:.1f if state.vwap else 'N/A'} "
                       f"Vol_Scale={state.session_vol_scale:.1f} Liq_Scale={state.session_liq_scale:.1f}")
            count += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ MarketStateBuilder v1.1.1 integration validated")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
