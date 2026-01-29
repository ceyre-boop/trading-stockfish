from engine.types import MarketState
from line_search_engine import LineSearchResult, SearchConfig, run_line_search


def _state(price: float, ma_short: float = 0.0, ma_long: float = 0.0) -> MarketState:
    return MarketState(
        current_price=price,
        ma_short=ma_short,
        ma_long=ma_long,
        momentum=price - ma_short,
        recent_returns=[0.01],
        volatility=0.1,
        liquidity=1.0,
    )


def test_single_step_search_matches_direct_policy_decision():
    current = _state(100, ma_short=90, ma_long=80)
    future = [_state(101, ma_short=90, ma_long=80)]
    result: LineSearchResult = run_line_search(
        current, future, SearchConfig(max_depth=1)
    )
    pv_actions = [n.action for n in result.principal_variation.nodes]
    assert pv_actions == ["ENTER"]


def test_multi_step_search_builds_all_lines_up_to_depth():
    current = _state(100, ma_short=90, ma_long=80)
    future = [_state(101, 90, 80), _state(102, 91, 81)]
    cfg = SearchConfig(max_depth=2, branching_limit=2)
    result = run_line_search(current, future, cfg)
    # With branching_limit=2, first two actions considered per node -> 2^2 lines
    assert len(result.all_lines) == 4


def test_ev_ranking_selects_higher_pnl_when_eval_equal():
    current = _state(100, 90, 90)
    # Price rises then drops: Enter early yields more pnl than holding flat
    future = [_state(105, 90, 90), _state(104, 90, 90)]
    cfg = SearchConfig(max_depth=2, branching_limit=2, weight_pnl=0.8, weight_eval=0.2)
    result = run_line_search(current, future, cfg)
    actions = [n.action for n in result.principal_variation.nodes]
    assert actions[0] == "ENTER"


def test_ev_ranking_respects_evaluator_when_pnl_similar():
    current = _state(100, 99, 98)
    # Same prices, but evaluator higher when ma_short below price
    future = [_state(100, 95, 94), _state(100, 95, 94)]
    cfg = SearchConfig(max_depth=2, branching_limit=1, weight_pnl=0.3, weight_eval=0.7)
    result = run_line_search(current, future, cfg)
    assert result.principal_variation.nodes[0].evaluator_score > 0


def test_determinism_same_inputs_same_output():
    current = _state(100, 90, 80)
    future = [_state(101, 90, 80), _state(102, 91, 81)]
    cfg = SearchConfig(max_depth=2, branching_limit=3)
    r1 = run_line_search(current, future, cfg)
    r2 = run_line_search(current, future, cfg)
    assert (
        r1.principal_variation.combined_ev_metric
        == r2.principal_variation.combined_ev_metric
    )
    assert [n.action for n in r1.principal_variation.nodes] == [
        n.action for n in r2.principal_variation.nodes
    ]
