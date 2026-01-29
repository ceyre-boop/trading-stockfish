"""
Mock pipelines are disabled. Real data only.
"""

raise RuntimeError(
    "mock_deps.py is disabled: synthetic pipelines are forbidden. Use real data adapters instead."
)

import math
from statistics import mean, pstdev

from validation.regime_logic import (
    liquidity_modifier,
    trend_modifier,
    volatility_modifier,
)


class MockEventBus:
    def subscribe(self, callback):
        pass


class MockStateBuilder:
    def __init__(self):
        self.mid_history = []

    def _rolling_std(self, data, window):
        if len(data) < 2:
            return 0.0
        windowed = data[-window:]
        if len(windowed) < 2:
            return 0.0
        return pstdev(windowed)

    def _rolling_mean(self, data, window):
        if not data:
            return 0.0
        return mean(data[-window:])

    def _slope(self, data):
        n = len(data)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = mean(data)
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(data))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    def build(self, event):
        tick = event.get("tick", {})
        book = event.get("book", {})
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = bids[0][0] if bids else tick.get("price", 0.0)
        best_ask = asks[0][0] if asks else tick.get("price", 0.0)
        spread = max(best_ask - best_bid, 1e-6)
        depth = 0.0
        for lvl in bids[:2]:
            depth += lvl[1]
        for lvl in asks[:2]:
            depth += lvl[1]
        depth = max(depth, 1e-6)
        mid = (best_bid + best_ask) / 2.0
        self.mid_history.append(mid)
        vol = self._rolling_std(self.mid_history, 50)
        liquidity = 1.0 / (spread * depth)
        trend = self._slope(self.mid_history[-20:])

        price_lookback_20 = (
            self.mid_history[-20]
            if len(self.mid_history) >= 20
            else self.mid_history[0]
        )
        rolling_mean_20 = self._rolling_mean(self.mid_history, 20)
        momentum = mid - price_lookback_20
        mean_rev = -(mid - rolling_mean_20)
        buy_volume = tick.get("buy_volume", tick.get("volume", 1))
        sell_volume = tick.get("sell_volume", tick.get("volume", 1))
        ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-9)

        # Regimes
        v1, v2 = 0.05, 0.15
        l1, l2 = 0.02, 0.005
        t1 = 0.0
        vol_regime = "LOW" if vol < v1 else "NORMAL" if vol < v2 else "HIGH"
        liq_regime = (
            "ROBUST" if liquidity > l1 else "NORMAL" if liquidity > l2 else "FRAGILE"
        )
        trend_regime = "UP" if trend > t1 else "DOWN" if trend < -t1 else "SIDEWAYS"

        factors = {
            "momentum": momentum,
            "mean_reversion": mean_rev,
            "orderflow_imbalance": ofi,
            "volatility_factor": vol,
            "liquidity_factor": liquidity,
        }

        market_state = {
            "timestamp": tick.get("timestamp"),
            "price": mid,
            "spread": spread,
            "volume": tick.get("volume", 1),
            "volatility": vol,
            "liquidity": liquidity,
            "trend": trend,
            "regime": {
                "volatility": vol_regime,
                "liquidity": liq_regime,
                "trend": trend_regime,
            },
            "factors": factors,
            "tick": tick,
            "book": book,
        }
        return market_state


class MockEvaluator:
    def evaluate(self, state):
        factors = state["factors"]
        weights = {
            "momentum": 0.3,
            "mean_reversion": 0.2,
            "orderflow_imbalance": 0.2,
            "volatility_factor": 0.15,
            "liquidity_factor": 0.15,
        }

        def _norm(x: float) -> float:
            return math.tanh(x)

        normalized_factors = {k: _norm(v) for k, v in factors.items()}
        factor_alignment = sum(
            normalized_factors.get(k, 0.0) * w for k, w in weights.items()
        )

        regimes = state.get("regime", {})
        v_mod = volatility_modifier(regimes.get("volatility"))
        l_mod = liquidity_modifier(regimes.get("liquidity"))
        t_mod = trend_modifier(regimes.get("trend"))
        base_conf = 0.6 + min(0.35, abs(factor_alignment))
        confidence = max(0.0, min(1.0, base_conf * v_mod * l_mod * t_mod))

        return {
            "score": factor_alignment,
            "confidence": confidence,
            "factors": factors,
            "normalized_factors": normalized_factors,
            "factor_alignment": factor_alignment,
            "factor_weights": weights,
            "regime_modifiers": {
                "volatility": v_mod,
                "liquidity": l_mod,
                "trend": t_mod,
            },
        }

    def generate_candidates(self, state, evaluation):
        factor_alignment = evaluation.get("factor_alignment", 0.0)
        action = "ENTER_LONG" if factor_alignment >= 0 else "ENTER_SHORT"
        return [
            {
                "action": action,
                "size": 0.02,
                "factor_alignment": factor_alignment,
                "confidence": evaluation.get("confidence", 0.0),
                "regime_context": state.get("regime", {}),
            }
        ]


class MockMetaPolicy:
    def apply(self, candidates, state):
        return candidates


class MockAdaptiveWeighting:
    def apply(self, actions, state):
        evaluation = {
            "factor_alignment": (
                actions[0].get("factor_alignment", 0.0) if actions else 0.0
            ),
            "confidence": actions[0].get("confidence", 0.0) if actions else 0.0,
            "factors": state.get("factors", {}),
        }
        policy = MockPolicy()
        decision = policy.decide(state, evaluation)
        return [decision]


class MockPolicy:
    def decide(self, state, evaluation):
        regimes = state.get("regime", {}) if isinstance(state, dict) else {}
        v_mod = volatility_modifier(regimes.get("volatility"))
        l_mod = liquidity_modifier(regimes.get("liquidity"))
        t_mod = trend_modifier(regimes.get("trend"))

        factor_alignment = evaluation.get("factor_alignment", 0.0)
        confidence = evaluation.get("confidence", 0.0)
        base_size = 0.02
        size = base_size * v_mod * l_mod * t_mod * max(confidence, 0.55)

        liq_regime = regimes.get("liquidity")
        if liq_regime == "FRAGILE" and confidence <= 0.85:
            return {
                "action": "FLAT",
                "size": 0.0,
                "reasoning": {
                    "note": "Liquidity fragile, confidence too low",
                    "confidence": confidence,
                },
                "regime_context": regimes,
                "factor_alignment": factor_alignment,
            }

        action = "BUY" if factor_alignment >= 0 else "SELL"
        size = max(size, 0.005)
        size = min(size, 0.05)

        reasoning = {
            "confidence": confidence,
            "factor_alignment": factor_alignment,
            "modifiers": {
                "volatility": v_mod,
                "liquidity": l_mod,
                "trend": t_mod,
            },
            "base_size": base_size,
            "final_size": size,
        }

        return {
            "action": action,
            "size": size,
            "reasoning": reasoning,
            "regime_context": regimes,
            "factor_alignment": factor_alignment,
        }


class MockExecutor:
    def __init__(self):
        self.open_trade = None
        self.trade_ticks = []

    def _mid(self, state):
        return state.get("price") or state.get("tick", {}).get("price", 0.0)

    def simulate(self, actions, state):
        # Choose first action
        action = actions[0] if actions else {"action": "HOLD", "size": 0}
        act = action.get("action", "HOLD")
        size = action.get("size", 0.0)
        mid = self._mid(state)
        last_trade_price = state.get("tick", {}).get("price", mid)

        # Entry
        if (
            self.open_trade is None
            and act in ["ENTER_LONG", "BUY", "buy", "ENTER_SHORT", "SELL", "sell"]
            and size > 0
        ):
            signed_size = size if act in ["ENTER_LONG", "BUY", "buy"] else -size
            entry_price = mid
            self.open_trade = {
                "entry_price": entry_price,
                "size": signed_size,
                "slippage": abs(entry_price - last_trade_price),
                "ticks": [],
            }
            self.trade_ticks = []
            return {"filled": True, "entry_price": entry_price, "size": signed_size}

        # Track during
        if self.open_trade is not None:
            self.trade_ticks.append({"mid": mid})

        # Auto-exit after 3 bars or explicit exit action
        should_exit = False
        if self.open_trade is not None:
            if len(self.trade_ticks) >= 3:
                should_exit = True
            if act in [
                "EXIT",
                "CLOSE",
                "SELL",
                "sell",
                "FLAT",
                "COVER",
                "TAKE_PROFIT",
                "STOP",
            ]:
                should_exit = True

        if should_exit and self.open_trade is not None:
            entry_price = self.open_trade["entry_price"]
            exit_price = mid
            size = self.open_trade["size"]
            pnl = (exit_price - entry_price) * size
            mae = (
                min((t["mid"] - entry_price) * size for t in self.trade_ticks)
                if self.trade_ticks
                else 0.0
            )
            mfe = (
                max((t["mid"] - entry_price) * size for t in self.trade_ticks)
                if self.trade_ticks
                else 0.0
            )
            bars_held = len(self.trade_ticks)
            slippage = self.open_trade.get("slippage", 0.0)
            transaction_cost = abs(size) * 0.25
            result = {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "filled": True,
                "size": size,
                "slippage": slippage,
                "transaction_cost": transaction_cost,
                "pnl": pnl - transaction_cost,
                "mae": mae,
                "mfe": mfe,
                "holding_period": bars_held,
            }
            self.open_trade = None
            self.trade_ticks = []
            return result

        return {"filled": False}


class MockPortfolioManager:
    def update(self, execution_result):
        pass


class MockDiagnostics:
    def emit(self, state, evaluation, actions, execution_result):
        regimes = state.get("regime", {}) if isinstance(state, dict) else {}
        policy_decision = actions[0] if actions else {}
        governance_decision = {
            "approved": True,
            "reason": "PASS",
            "regime": regimes,
            "confidence": (
                evaluation.get("confidence", 0.0)
                if isinstance(evaluation, dict)
                else 0.0
            ),
        }
        return {
            "market_state": state,
            "evaluator_output": evaluation,
            "policy_decision": policy_decision,
            "governance_decision": governance_decision,
            "execution_result": execution_result,
        }


def get_mock_deps():
    return {
        "event_bus": MockEventBus(),
        "state_builder": MockStateBuilder(),
        "evaluator": MockEvaluator(),
        "meta_policy": MockMetaPolicy(),
        "adaptive_weighting": MockAdaptiveWeighting(),
        "executor": MockExecutor(),
        "portfolio_manager": MockPortfolioManager(),
        "diagnostics": MockDiagnostics(),
    }
