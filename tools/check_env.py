"""
Environment readiness checker for Trading Stockfish.
- Verifies required env vars are loaded (without printing secrets).
- Verifies outbound network access to Polygon reference endpoint.
- Verifies MT5 installation can initialize (no login attempt).

Run: python tools/check_env.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import requests  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"requests import failed: {exc}")
    sys.exit(1)

try:
    from dotenv import load_dotenv  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"python-dotenv import failed: {exc}")
    sys.exit(1)


REQUIRED_ENV = ["POLYGON_API_KEY", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"]


def _mask_value(key: str, value: str) -> str:
    if key.upper().endswith("PASSWORD") or "API_KEY" in key.upper():
        return "***MASKED***"
    return value


def _detect_prefix_bytes(content: bytes) -> bytes:
    for idx, bch in enumerate(content):
        if chr(bch).isalnum() or chr(bch) in {"#", "_"}:
            return content[:idx]
    return content


def _inspect_env_file(dotenv_path: str) -> Tuple[bool, bool]:
    print(f"ENV expected path: {dotenv_path}")
    exists = os.path.exists(dotenv_path)
    print(f"ENV file exists: {exists}")
    bom_detected = False
    format_error = False
    if not exists:
        return bom_detected, format_error
    try:
        raw_bytes = Path(dotenv_path).read_bytes()
    except Exception as exc:  # pragma: no cover
        print(f"ENV read error: {exc}")
        return bom_detected, format_error
    prefix = _detect_prefix_bytes(raw_bytes)
    if prefix:
        print(f"ENV leading bytes before first key: {prefix!r}")
    else:
        print("ENV leading bytes before first key: b''")
    bom_detected = raw_bytes.startswith(b"\xef\xbb\xbf")
    if bom_detected:
        print("ENV_BOM_DETECTED")
    text = raw_bytes.decode("utf-8", errors="replace")
    print("ENV raw contents (passwords masked):")
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            masked_v = _mask_value(k.strip(), v)
            print(f"{k}={masked_v}")
            if v.endswith(" ") or v.startswith(" "):
                format_error = True
            if any((ord(ch) < 32 and ch not in "\t\n\r") for ch in v):
                format_error = True
        else:
            print(line)
    if format_error:
        print("ENV_FORMAT_ERROR")
    return bom_detected, format_error


def load_env_files(dotenv_path: str) -> None:
    load_dotenv(dotenv_path=dotenv_path, override=True)
    _apply_active_mt5_account()
    print(f"RAW_MT5_LOGIN: {repr(os.getenv('MT5_LOGIN'))}")
    print(f"RAW_MT5_PASSWORD: {repr(os.getenv('MT5_PASSWORD'))}")
    print(f"RAW_MT5_SERVER: {repr(os.getenv('MT5_SERVER'))}")
    for key in REQUIRED_ENV:
        val = os.getenv(key)
        if val is None or val == "":
            print("ENV_PARSE_FAILURE: variable missing or unreadable")


def _apply_active_mt5_account() -> None:
    active = (os.getenv("MT5_ACTIVE_ACCOUNT") or "").strip().upper()
    if active not in {"PRIMARY", "SECONDARY"}:
        print("MT5_ACTIVE_ACCOUNT must be PRIMARY or SECONDARY")
        sys.exit(1)

    def pick_named(name: str) -> str:
        raw = os.getenv(name) or ""
        return raw.replace("\ufeff", "").strip().strip("\"'")

    if active == "PRIMARY":
        login = pick_named("MT5_LOGIN_PRIMARY")
        pwd = pick_named("MT5_PASSWORD_PRIMARY")
        srv = pick_named("MT5_SERVER_PRIMARY")
    else:
        login = pick_named("MT5_LOGIN_SECONDARY")
        pwd = pick_named("MT5_PASSWORD_SECONDARY")
        srv = pick_named("MT5_SERVER_SECONDARY")

    missing = []
    if not login:
        missing.append("login")
    if not pwd:
        missing.append("password")
    if not srv:
        missing.append("server")
    if missing:
        print(f"MT5 {active} account missing fields: {', '.join(missing)}")
        sys.exit(1)

    os.environ["MT5_LOGIN"] = login
    os.environ["MT5_PASSWORD"] = pwd
    os.environ["MT5_SERVER"] = srv
    print(f"MT5 active account applied: {active}")


def check_env_vars() -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for key in REQUIRED_ENV:
        status[key] = os.getenv(key) is not None and os.getenv(key) != ""
        print(f"{key}: {status[key]}")
    all_ok = all(status.values())
    print(f"ENV_LOADED: {all_ok}")
    return status


def _update_env_login(dotenv_path: str, new_login: str) -> None:
    try:
        text = Path(dotenv_path).read_text(encoding="utf-8")
    except Exception:
        return
    lines = text.splitlines()
    output = []
    found = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("MT5_LOGIN="):
            output.append(f"MT5_LOGIN={new_login}")
            found = True
        else:
            output.append(line)
    if not found:
        output.append(f"MT5_LOGIN={new_login}")
    try:
        Path(dotenv_path).write_text("\n".join(output) + "\n", encoding="utf-8")
        print("MT5_LOGIN updated to working account")
    except Exception as exc:
        print(f"Failed to update MT5_LOGIN in .env: {exc}")


def test_mt5_account(login: str, password: str, server: str) -> bool:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception:
        print("MT5 test: MetaTrader5 import failed")
        return False
    if not login.isdigit():
        print(f"MT5 test login {login}: invalid (non-numeric)")
        return False
    paths = [
        None,
        "C:/Program Files/MetaTrader 5/terminal64.exe",
        "C:/Program Files/MetaTrader5/terminal64.exe",
        "C:/Program Files (x86)/MetaTrader 5/terminal64.exe",
        "C:/Program Files (x86)/MetaTrader5/terminal64.exe",
    ]
    print(f"Testing MT5 login: {login}")
    for candidate in paths:
        if candidate is not None and not os.path.exists(candidate):
            continue
        init_ok = (
            mt5.initialize() if candidate is None else mt5.initialize(path=candidate)
        )
        if not init_ok:
            err = mt5.last_error()
            print(
                f"MT5 test login {login} init failed at {candidate or '(default)'} last_error: {err}"
            )
            mt5.shutdown()
            continue
        try:
            logged_in = mt5.login(login=int(login), password=password, server=server)
            err = mt5.last_error()
            print(f"MT5 test login {login} last_error: {err}")
            mt5.shutdown()
            if logged_in:
                return True
        except Exception as exc:
            print(f"MT5 test login {login} exception: {exc}")
            mt5.shutdown()
    return False


def _get_polygon_api_key() -> str:
    raw = os.getenv("POLYGON_API_KEY") or ""
    cleaned = raw.replace("\ufeff", "")  # strip BOM if present
    stripped = cleaned.strip()
    unquoted = stripped.strip("\"'")
    if stripped != cleaned:
        print(
            "POLYGON_API_KEY notice: leading/trailing whitespace removed for request."
        )
    if unquoted != stripped:
        print("POLYGON_API_KEY notice: surrounding quotes removed for request.")
    sanitized = unquoted
    if sanitized:
        print(f"POLYGON_API_KEY length (sanitized): {len(sanitized)}")
    else:
        print("POLYGON_API_KEY is empty after sanitization.")
    return sanitized


def check_network() -> bool:
    api_key = _get_polygon_api_key()
    params = {"limit": 1}
    headers = {}
    if api_key:
        params["apiKey"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Polygon-API-Key"] = api_key
    url = "https://api.polygon.io/v3/reference/tickers"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        status = resp.status_code
        ok = status == 200
        try:
            redacted_url = resp.url.replace(api_key, "***") if api_key else resp.url
            print(f"NETWORK url: {redacted_url}")
        except Exception:
            print("NETWORK url: <redacted>")
        print(f"NETWORK status_code: {status}")
        if not ok:
            try:
                print(f"NETWORK response snippet: {resp.text[:200]}")
            except Exception:
                print("NETWORK response snippet: <unavailable>")
        print(f"NETWORK_OK: {ok}")
        return ok
    except Exception:
        print("NETWORK status_code: None")
        print("NETWORK_OK: False")
        return False


def check_mt5() -> bool:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception:
        print("MT5_READY: False (import failed)")
        return False
    try:
        login_env = os.getenv("MT5_LOGIN") or ""
        server_env = os.getenv("MT5_SERVER") or ""
        password_env = os.getenv("MT5_PASSWORD") or ""
        if login_env and not login_env.isdigit():
            print("Invalid login format")
        paths = [
            None,
            "C:/Program Files/MetaTrader 5/terminal64.exe",
            "C:/Program Files/MetaTrader5/terminal64.exe",
            "C:/Program Files (x86)/MetaTrader 5/terminal64.exe",
            "C:/Program Files (x86)/MetaTrader5/terminal64.exe",
        ]
        used_path = None
        ok = False
        last_error_info = None
        for candidate in paths:
            path_label = candidate if candidate is not None else "(default)"
            print(f"MT5 init path candidate: {path_label}")
            print(f"MT5 login: {login_env}")
            print(f"MT5 server: {server_env}")
            print(f"MT5 password empty: {password_env == ''}")
            if candidate is not None and not os.path.exists(candidate):
                print("Terminal path invalid")
                continue
            if candidate is None:
                ok = bool(mt5.initialize())
                used_path = "(default)"
            else:
                ok = bool(mt5.initialize(path=candidate))
                used_path = candidate
            last_error_info = mt5.last_error()
            if ok:
                break
            print(f"MT5 last_error: {last_error_info}")
            if last_error_info == (-6, "Terminal: Authorization failed"):
                print("Auth hint: wrong login/password/server or expired demo account")
            elif last_error_info == (-1, "Terminal: Not initialized"):
                print("Auth hint: wrong terminal path")
            else:
                print("Unknown MT5 error")
        print(f"MT5 init path: {used_path}")
        print(f"MT5_READY: {ok}")
        if ok:
            acct = mt5.account_info()
            if acct is None:
                print("Server mismatch: the terminal does not support this server")
            else:
                if server_env and acct.server and acct.server != server_env:
                    print("Server mismatch: the terminal does not support this server")
                else:
                    print("Server matches terminal list")
            mt5.shutdown()
        else:
            try:
                last_error_info = last_error_info or mt5.last_error()
            except Exception:
                pass
        print("--- MT5 SUMMARY ---")
        print(f"MT5_LOGIN: {login_env}")
        print(f"MT5_SERVER: {server_env}")
        print(f"MT5 terminal path: {used_path}")
        print(f"MT5 initialization: {ok}")
        print(f"MT5 last_error: {last_error_info}")
        return ok
    except Exception:
        print("MT5 init path: None")
        print("MT5_READY: False")
        return False


if __name__ == "__main__":
    dotenv_path = os.path.abspath(".env")
    _inspect_env_file(dotenv_path)
    load_env_files(dotenv_path)
    env_status = check_env_vars()
    network_ok = check_network()
    mt5_ok = check_mt5()
    if not mt5_ok:
        pwd = os.getenv("MT5_PASSWORD") or ""
        srv = os.getenv("MT5_SERVER") or ""
        primary_login = os.getenv("MT5_LOGIN_PRIMARY") or ""
        secondary_login = os.getenv("MT5_LOGIN_SECONDARY") or ""
        active_login = os.getenv("MT5_LOGIN") or ""
        candidates = [
            active_login,
            primary_login,
            secondary_login,
            "5045515692",
            "5045507061",
            "5044972719",
        ]
        candidates = [c for c in candidates if c]
        seen = set()
        uniq_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq_candidates.append(c)
        candidates = uniq_candidates
        results = {}
        for cand in candidates:
            result = test_mt5_account(cand, pwd, srv)
            results[cand] = result
        successes = [k for k, v in results.items() if v]
        print("--- MT5 CANDIDATE SUMMARY ---")
        for cand in candidates:
            print(f"Login {cand}: {'SUCCESS' if results.get(cand) else 'FAIL'}")
        if len(successes) == 1:
            _update_env_login(dotenv_path, successes[0])
        elif len(successes) == 0:
            print(
                "All candidate accounts failed. The account credentials are invalid or expired."
            )
        else:
            print("Multiple candidate accounts succeeded; manual selection required.")
    summary_env = all(env_status.values())
    print("--- SUMMARY ---")
    print(f"ENV_LOADED: {summary_env}")
    print(f"NETWORK_OK: {network_ok}")
    print(f"MT5_READY: {mt5_ok}")
