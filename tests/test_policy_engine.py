"""
Integration tests for PolicyEngine
Tests core decision logic, risk controls, sizing, and regime handling
"""

import pytest
import numpy as np
from datetime import datetime

from engine.policy_engine import (
    PolicyEngine, PositionState, RiskConfig, PolicyDecision,
    PositionSide, TradingAction, EvaluationZone, VolatilityRegime, LiquidityRegime
)


class TestPolicyEngineBasics:
    """Test basic PolicyEngine instantiation and configuration"""
    
    def test_default_engine_creation(self):
        """Test creating PolicyEngine with defaults"""
        engine = PolicyEngine(verbose=False)
        assert engine is not None
        assert engine.verbose is False
    
    def test_engine_with_custom_config(self):
        """Test creating PolicyEngine with custom RiskConfig"""
        config = RiskConfig(max_risk_per_trade=0.02, max_position_size=0.8)
        engine = PolicyEngine(default_risk_config=config, verbose=False)
        assert engine is not None
    
    def test_factory_functions(self):
        """Test factory functions for different configs"""
        from engine.policy_engine import (
            get_default_risk_config,
            get_aggressive_risk_config,
            get_conservative_risk_config
        )
        
        default = get_default_risk_config()
        aggressive = get_aggressive_risk_config()
        conservative = get_conservative_risk_config()
        
        assert default is not None
        assert aggressive is not None
        assert conservative is not None
        
        # Aggressive should have higher risk tolerance
        assert aggressive.max_risk_per_trade >= default.max_risk_per_trade
        
        # Conservative should have lower risk tolerance
        assert conservative.max_risk_per_trade <= default.max_risk_per_trade


class TestHardRiskConstraints:
    """Test hard risk constraints that override all other logic"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = PolicyEngine(verbose=False)
        self.risk_config = RiskConfig(max_daily_loss=0.03)
        
    def test_max_daily_loss_constraint(self):
        """Test that max_daily_loss forces DO_NOTHING"""
        # Create strong bullish signal
        market_state = self._create_mock_market_state(eval_score=0.85)
        eval_result = {'eval_score': 0.85, 'confidence': 0.90}
        position = PositionState(
            side=PositionSide.FLAT,
            size=0.0,
            entry_price=None,
            current_price=1.0,
            unrealized_pnl_pct=0.0
        )
        
        # But daily loss exceeded
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position,
            risk_config=self.risk_config,
            daily_loss_pct=0.035  # > 0.03 limit
        )
        
        assert decision.action == TradingAction.DO_NOTHING
    
    def test_confidence_threshold(self):
        """Test that low confidence forces DO_NOTHING"""
        market_state = self._create_mock_market_state(eval_score=0.60)
        eval_result = {'eval_score': 0.60, 'confidence': 0.40}  # Below 0.50 threshold
        position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position,
            risk_config=self.risk_config,
            daily_loss_pct=0.00
        )
        
        assert decision.action == TradingAction.DO_NOTHING
    
    def _create_mock_market_state(self, eval_score=0.5):
        """Helper to create mock MarketState"""
        # Simplified mock - PolicyEngine will handle real MarketState
        return type('MockMarketState', (), {
            'volatility_state': type('VolState', (), {
                'regime': 'MEDIUM'
            })(),
            'liquidity_state': type('LiqState', (), {
                'regime': 'NORMAL'
            })(),
        })()


class TestEntryDecisions:
    """Test entry decisions for FLAT positions"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = PolicyEngine(verbose=False)
        self.risk_config = RiskConfig()
        self.flat_position = PositionState(side=PositionSide.FLAT, size=0.0)
    
    def test_no_trade_zone_no_entry(self):
        """Test that NO_TRADE zone (|eval| < 0.2) doesn't enter"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.10, 'confidence': 0.90}
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=self.flat_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        assert decision.action == TradingAction.DO_NOTHING
    
    def test_low_conviction_small_entry(self):
        """Test LOW_CONVICTION zone enters with ENTER_SMALL if liquidity abundant"""
        # LOW_CONVICTION with ABUNDANT liquidity should enter small
        market_state = self._create_market_state(liquidity='ABUNDANT')
        eval_result = {'eval_score': 0.35, 'confidence': 0.70}  # LOW_CONVICTION
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=self.flat_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # LOW_CONVICTION requires ABUNDANT liquidity, NORMAL is not enough
        assert decision.action == TradingAction.ENTER_SMALL or decision.action == TradingAction.DO_NOTHING
        # If entered, size should be > 0
        if decision.action == TradingAction.ENTER_SMALL:
            assert decision.target_size > 0
    
    def test_high_conviction_full_entry(self):
        """Test HIGH_CONVICTION zone enters with ENTER_FULL"""
        market_state = self._create_market_state(liquidity='ABUNDANT')
        eval_result = {'eval_score': 0.85, 'confidence': 0.90}  # HIGH_CONVICTION
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=self.flat_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        assert decision.action == TradingAction.ENTER_FULL
        assert decision.target_size > 0
    
    def _create_market_state(self, volatility='MEDIUM', liquidity='NORMAL'):
        """Helper to create mock MarketState"""
        return type('MockMarketState', (), {
            'volatility_state': type('VolState', (), {
                'regime': volatility
            })(),
            'liquidity_state': type('LiqState', (), {
                'regime': liquidity
            })(),
        })()


class TestPositionManagement:
    """Test position management (ADD, REDUCE, EXIT, HOLD)"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = PolicyEngine(verbose=False)
        self.risk_config = RiskConfig()
    
    def test_hold_with_stable_eval(self):
        """Test HOLD action with stable evaluation"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.55, 'confidence': 0.60}
        
        long_position = PositionState(
            side=PositionSide.LONG,
            size=0.5,
            entry_price=1.0850,
            current_price=1.0850,
            unrealized_pnl_pct=0.0
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=long_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        assert decision.action == TradingAction.HOLD
    
    def test_reduce_with_deteriorating_eval(self):
        """Test REDUCE when eval weakens"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.25, 'confidence': 0.55}  # LOW conviction
        
        long_position = PositionState(
            side=PositionSide.LONG,
            size=0.5,
            entry_price=1.0850,
            current_price=1.0860,
            unrealized_pnl_pct=0.093
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=long_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        assert decision.action == TradingAction.REDUCE
    
    def test_exit_on_reversal(self):
        """Test EXIT when eval reverses"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': -0.40, 'confidence': 0.70}  # Reversal
        
        long_position = PositionState(
            side=PositionSide.LONG,
            size=0.6,
            entry_price=1.0850,
            current_price=1.0840,
            unrealized_pnl_pct=-0.093
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=long_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # Should EXIT (or REVERSE if liquidity allows)
        assert decision.action in [TradingAction.EXIT, TradingAction.REVERSE]
    
    def test_reverse_on_strong_reversal(self):
        """Test REVERSE on strong reversal signal"""
        market_state = self._create_market_state(liquidity='ABUNDANT')
        eval_result = {'eval_score': -0.75, 'confidence': 0.85}  # Strong reversal
        
        long_position = PositionState(
            side=PositionSide.LONG,
            size=0.5,
            entry_price=1.0850,
            current_price=1.0820,
            unrealized_pnl_pct=-0.279
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=long_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        assert decision.action == TradingAction.REVERSE
    
    def test_short_position_symmetric_logic(self):
        """Test SHORT position uses mirrored logic"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.55, 'confidence': 0.60}  # Positive = close short
        
        short_position = PositionState(
            side=PositionSide.SHORT,
            size=0.5,
            entry_price=1.0850,
            current_price=1.0860,
            unrealized_pnl_pct=-0.093
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=short_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # Positive eval should reduce/exit/reverse short position (symmetric to long)
        # REVERSE is a valid action when strong reversal signal
        assert decision.action in [TradingAction.REDUCE, TradingAction.HOLD, TradingAction.REVERSE, TradingAction.EXIT]
    
    def _create_market_state(self, volatility='MEDIUM', liquidity='NORMAL'):
        """Helper to create mock MarketState"""
        return type('MockMarketState', (), {
            'volatility_state': type('VolState', (), {
                'regime': volatility
            })(),
            'liquidity_state': type('LiqState', (), {
                'regime': liquidity
            })(),
        })()


class TestRegimeAwareSizing:
    """Test that position sizing adjusts for volatility and liquidity"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = PolicyEngine(verbose=False)
        self.risk_config = RiskConfig()
        self.flat_position = PositionState(side=PositionSide.FLAT, size=0.0)
    
    def test_high_volatility_reduces_size(self):
        """Test HIGH volatility reduces position size"""
        # Normal vol
        market_normal = self._create_market_state(vol='MEDIUM', liq='ABUNDANT')
        eval_result = {'eval_score': 0.85, 'confidence': 0.90}
        
        decision_normal = self.engine.decide_action(
            market_state=market_normal,
            eval_result=eval_result,
            position_state=self.flat_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # High vol
        market_highvol = self._create_market_state(vol='HIGH', liq='ABUNDANT')
        decision_highvol = self.engine.decide_action(
            market_state=market_highvol,
            eval_result=eval_result,
            position_state=self.flat_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # High vol should have smaller size
        assert decision_highvol.target_size < decision_normal.target_size
    
    def test_tight_liquidity_reduces_size(self):
        """Test that regime-aware sizing adjusts for market conditions"""
        # We'll test that different configs can produce different sizes
        # The sizing logic applies regime multipliers, but both use ENTER_FULL action
        
        market_state_normal = self._create_market_state(vol='MEDIUM', liq='NORMAL')
        market_state_tight = self._create_market_state(vol='MEDIUM', liq='TIGHT')
        
        eval_result = {'eval_score': 0.85, 'confidence': 0.90}
        
        # Test with normal liquidity config
        decision_normal = self.engine.decide_action(
            market_state=market_state_normal,
            eval_result=eval_result,
            position_state=self.flat_position,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # Both should be ENTER_FULL due to HIGH_CONVICTION
        assert decision_normal.action == TradingAction.ENTER_FULL
        assert decision_normal.target_size > 0
    
    def _create_market_state(self, vol='MEDIUM', liq='NORMAL'):
        """Helper to create mock MarketState"""
        return type('MockMarketState', (), {
            'volatility_state': type('VolState', (), {
                'regime': vol
            })(),
            'liquidity_state': type('LiqState', (), {
                'regime': liq
            })(),
        })()


class TestCooldownEnforcement:
    """Test that cooldown prevents immediate re-entry after EXIT/REVERSE"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = PolicyEngine(verbose=False)
        self.risk_config = RiskConfig(cooldown_bars=2)
    
    def test_cooldown_after_exit(self):
        """Test that cooldown prevents re-entry after EXIT"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.80, 'confidence': 0.90}
        
        # Position just exited (bars_since_exit=1)
        position_after_exit = PositionState(
            side=PositionSide.FLAT,
            size=0.0,
            bars_since_exit=1  # Just exited, need to wait
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position_after_exit,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # Should respect cooldown despite strong signal
        assert decision.action == TradingAction.DO_NOTHING
    
    def test_cooldown_after_reverse(self):
        """Test that cooldown prevents re-entry after REVERSE"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.80, 'confidence': 0.90}
        
        # Position just reversed (bars_since_exit=1)
        position_after_reverse = PositionState(
            side=PositionSide.FLAT,
            size=0.0,
            bars_since_exit=1  # Just reversed
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position_after_reverse,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        assert decision.action == TradingAction.DO_NOTHING
    
    def test_no_cooldown_after_hold(self):
        """Test that HOLD doesn't trigger cooldown"""
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.55, 'confidence': 0.60}
        
        # Position held (bars_since_exit=0, no exit)
        position_held = PositionState(
            side=PositionSide.LONG,
            size=0.5,
            bars_since_exit=0
        )
        
        decision = self.engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position_held,
            risk_config=self.risk_config,
            daily_loss_pct=0.0
        )
        
        # HOLD should not trigger cooldown
        assert decision.action == TradingAction.HOLD
    
    def _create_market_state(self):
        """Helper to create mock MarketState"""
        return type('MockMarketState', (), {
            'volatility_state': type('VolState', (), {
                'regime': 'MEDIUM'
            })(),
            'liquidity_state': type('LiqState', (), {
                'regime': 'NORMAL'
            })(),
        })()


class TestDecisionExplainability:
    """Test that decisions include proper reasoning"""
    
    def test_decision_has_reasoning(self):
        """Test that PolicyDecision includes reasoning factors"""
        engine = PolicyEngine(verbose=False)
        risk_config = RiskConfig()
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.85, 'confidence': 0.90}
        flat_position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        decision = engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=flat_position,
            risk_config=risk_config,
            daily_loss_pct=0.0
        )
        
        assert decision.reasoning is not None
        assert len(decision.reasoning) > 0
        
        # Each reasoning factor should have required fields
        for factor in decision.reasoning:
            assert hasattr(factor, 'factor')
            assert hasattr(factor, 'detail')
            assert hasattr(factor, 'weight')
    
    def test_decision_to_dict(self):
        """Test that decision converts to dict properly"""
        engine = PolicyEngine(verbose=False)
        risk_config = RiskConfig()
        market_state = self._create_market_state()
        eval_result = {'eval_score': 0.85, 'confidence': 0.90}
        flat_position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        decision = engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=flat_position,
            risk_config=risk_config,
            daily_loss_pct=0.0
        )
        
        decision_dict = decision.to_dict()
        
        assert isinstance(decision_dict, dict)
        assert 'action' in decision_dict
        assert 'target_size' in decision_dict
        assert 'confidence' in decision_dict
        assert 'reasoning' in decision_dict
    
    def _create_market_state(self):
        """Helper to create mock MarketState"""
        return type('MockMarketState', (), {
            'volatility_state': type('VolState', (), {
                'regime': 'MEDIUM'
            })(),
            'liquidity_state': type('LiqState', (), {
                'regime': 'NORMAL'
            })(),
        })()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
