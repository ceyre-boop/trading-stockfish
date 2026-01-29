import pandas as pd

from analytics.replay_day import ReplayEngine


def test_replay_engine_emits_amd_state():
    data = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(60)],
            "high": [100.2 + i * 0.1 for i in range(60)],
            "low": [99.8 + i * 0.1 for i in range(60)],
            "close": [100.1 + i * 0.1 for i in range(60)],
            "volume": [1000 + i for i in range(60)],
        }
    )
    engine = ReplayEngine(symbol="TEST", data=data, verbose=False)
    snapshot = engine.step()
    assert snapshot is not None
    assert "amd_state" in snapshot.market_state
    assert "amd_regime" in snapshot.market_state
