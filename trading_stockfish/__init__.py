"""trading-stockfish: scientific benchmarking engine for intraday futures trading."""

from .engine import BenchmarkEngine, EngineConfig, RunResult

__version__ = "0.1.0"
__all__ = ["BenchmarkEngine", "EngineConfig", "RunResult", "__version__"]
