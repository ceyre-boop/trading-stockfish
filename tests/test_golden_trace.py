import json
import tempfile

from golden_trace import (
    GoldenTraceCollector,
    compare_golden_traces,
    load_golden_trace,
    save_golden_trace,
)
from tournament_harness import (
    compare_against_golden,
    generate_golden_trace,
    sample_market_states,
)


def test_golden_trace_roundtrip(tmp_path):
    states = sample_market_states()
    collector = GoldenTraceCollector()
    trace = collector.collect(states)

    out_path = tmp_path / "golden_trace.json"
    save_golden_trace(str(out_path), trace)
    loaded = load_golden_trace(str(out_path))

    assert compare_golden_traces(trace, loaded)[0]


def test_golden_trace_stability():
    states = sample_market_states()
    collector = GoldenTraceCollector()
    trace1 = collector.collect(states)
    trace2 = collector.collect(states)

    assert compare_golden_traces(trace1, trace2)[0]


def test_golden_trace_tournament_integration(tmp_path):
    states = sample_market_states()
    golden_path = tmp_path / "golden_trace.json"
    trace = generate_golden_trace(states)
    save_golden_trace(str(golden_path), trace)

    compare_against_golden(str(golden_path), states)

    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == len(trace)
