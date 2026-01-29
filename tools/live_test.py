"""End-to-end live MT5 + Trading Stockfish validation.

Run: python tools/live_test.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(f"pandas import failed: {exc}")
    sys.exit(1)

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"MetaTrader5 import failed: {exc}")
    sys.exit(1)

try:
    from engine.core import TradingStockfish
    from engine.data import MarketState
except Exception as exc:  # pragma: no cover
    print(f"Engine import failed: {exc}")
    sys.exit(1)

try:
    from tools import check_env
except Exception as exc:  # pragma: no cover
    print(f"check_env import failed: {exc}")
    sys.exit(1)


def load_env() -> None:
    """Load .env with the same active-account logic as check_env."""
    dotenv_path = os.path.abspath(".env")
    check_env.load_env_files(dotenv_path)


def init_mt5(login: int, password: str, server: str) -> bool:
    """Initialize MT5 and log in with provided credentials."""
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    print(f"MT5 last_error after init: {mt5.last_error()}")
    if not mt5.login(login=login, password=password, server=server):
        print(f"MT5 login failed: {mt5.last_error()}")
        return False
    return True


def fetch_tick(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"No tick data for {symbol}")
        return None
    print("=== LIVE TICK ===")
    print(f"Symbol: {symbol}")
    print(f"Bid: {tick.bid}")
    print(f"Ask: {tick.ask}")
    print(f"Last: {tick.last}")
    print(f"Time: {tick.time}")
    return tick


def fetch_bars(symbol: str, count: int = 100) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
    if rates is None or len(rates) == 0:
        print(f"No bar data returned for {symbol}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    print("=== PRICE ACTION (M1) ===")
    print(df.tail())
    return df


def evaluate_latest_bar(df: pd.DataFrame) -> None:
    if df.empty:
        print("Cannot evaluate: bar DataFrame is empty")
        return
    last_bar = df.iloc[-1]
    try:
        state = MarketState(
            open=last_bar["open"],
            high=last_bar["high"],
            low=last_bar["low"],
            close=last_bar["close"],
            volume=last_bar.get("tick_volume", last_bar.get("real_volume", 0)),
            timestamp=last_bar["time"],
        )
    except Exception as exc:
        print(f"Failed to build MarketState: {exc}")
        return
    try:
        engine = TradingStockfish()
        result = engine.evaluate(state)
        print("=== STOCKFISH EVALUATION ===")
        print(result)
    except Exception as exc:
        print(f"Engine evaluation failed: {exc}")


def main() -> None:
    load_env()
    login = os.getenv("MT5_LOGIN") or ""
    password = os.getenv("MT5_PASSWORD") or ""
    server = os.getenv("MT5_SERVER") or ""
    active = (os.getenv("MT5_ACTIVE_ACCOUNT") or "PRIMARY").strip().upper()
    print("=== ACTIVE MT5 ACCOUNT ===")
    print(f"Active selection: {active}")
    print(f"Login: {login}")
    print(f"Server: {server}")

    if not (login and password and server):
        print("Missing MT5 credentials; aborting.")
        sys.exit(1)

    try:
        login_int = int(login)
    except Exception:
        print(f"Invalid MT5 login (non-numeric): {login}")
        sys.exit(1)

    if not init_mt5(login_int, password, server):
        sys.exit(1)

    try:
        term_info = mt5.terminal_info()
        acct_info = mt5.account_info()
        print("=== MT5 TERMINAL INFO ===")
        print(term_info)
        print("=== MT5 ACCOUNT INFO ===")
        print(acct_info)
    except Exception as exc:
        print(f"Failed to read MT5 terminal/account info: {exc}")

    fetch_tick("EURUSD")
    df = fetch_bars("EURUSD", 100)
    evaluate_latest_bar(df)

    mt5.shutdown()


if __name__ == "__main__":
    main()
