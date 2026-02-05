from engine.decision_frame import DecisionFrame
from engine.mcr_pathgen import generate_price_paths


def test_determinism_same_seed_same_paths():
    frame = DecisionFrame()
    paths1 = generate_price_paths(frame, n_paths=3, horizon_bars=5, seed=42)
    paths2 = generate_price_paths(frame, n_paths=3, horizon_bars=5, seed=42)

    assert len(paths1) == len(paths2)
    for p1, p2 in zip(paths1, paths2):
        assert p1.prices == p2.prices
        assert p1.metadata == p2.metadata


def test_different_seed_changes_paths():
    frame = DecisionFrame()
    paths1 = generate_price_paths(frame, n_paths=2, horizon_bars=4, seed=1)
    paths2 = generate_price_paths(frame, n_paths=2, horizon_bars=4, seed=2)
    assert any(p1.prices != p2.prices for p1, p2 in zip(paths1, paths2))


def test_path_shapes_and_metadata():
    frame = DecisionFrame()
    n_paths = 2
    horizon = 6
    paths = generate_price_paths(frame, n_paths=n_paths, horizon_bars=horizon, seed=7)

    assert len(paths) == n_paths
    for idx, path in enumerate(paths):
        assert len(path.prices) == horizon + 1
        assert path.metadata["seed"] == 7
        assert path.metadata["path_index"] == idx
        # Prices should evolve (not all equal)
        assert len(set(path.prices)) > 1
