from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

# --------------------------- filter models ---------------------------


@dataclass(frozen=True)
class DecisionsFilter:
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    symbols: Optional[Sequence[str]] = None
    session_regimes: Optional[Sequence[str]] = None
    macro_regimes: Optional[Sequence[str]] = None
    timeframes: Optional[Sequence[str]] = None
    policy_versions: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class StatsFilter:
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    regimes: Optional[Sequence[str]] = None
    feature_names: Optional[Sequence[str]] = None
    run_ids: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class PolicyFilter:
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    policy_versions: Optional[Sequence[str]] = None
    sources: Optional[Sequence[str]] = None
    engine_versions: Optional[Sequence[str]] = None


# --------------------------- helpers ---------------------------


def _date_key(ts: str) -> Optional[str]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y%m%d")
    except Exception:
        return None


def _within_range(ts: str, start: Optional[date], end: Optional[date]) -> bool:
    if start is None and end is None:
        return True
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except Exception:
        return False
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


def _load_parquet_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _maybe_filter(
    df: pd.DataFrame, column: str, values: Optional[Sequence[str]]
) -> pd.DataFrame:
    if values is None or column not in df.columns:
        return df
    return df[df[column].isin(values)].reset_index(drop=True)


# --------------------------- loaders ---------------------------


def load_decisions(filter: DecisionsFilter) -> pd.DataFrame:
    base = Path("storage/decisions")
    files = sorted(base.glob("decisions_*.parquet")) if base.exists() else []

    # Narrow files by date range using filename partition when possible
    def _file_in_range(p: Path) -> bool:
        if filter.start_date is None and filter.end_date is None:
            return True
        stem = p.stem  # decisions_YYYYMMDD
        try:
            part = stem.split("_")[1]
            file_date = datetime.strptime(part, "%Y%m%d").date()
        except Exception:
            return True
        if filter.start_date and file_date < filter.start_date:
            return False
        if filter.end_date and file_date > filter.end_date:
            return False
        return True

    files = [p for p in files if _file_in_range(p)]
    df = _load_parquet_files(files)
    if df.empty:
        return df

    # Apply row-level filters
    if "timestamp_utc" in df.columns and (filter.start_date or filter.end_date):
        mask = df["timestamp_utc"].apply(
            lambda ts: _within_range(str(ts), filter.start_date, filter.end_date)
        )
        df = df[mask]

    df = _maybe_filter(df, "symbol", filter.symbols)
    df = _maybe_filter(df, "session_regime", filter.session_regimes)
    df = _maybe_filter(df, "timeframe", filter.timeframes)
    df = _maybe_filter(df, "policy_version", filter.policy_versions)

    if filter.macro_regimes and "macro_regimes" in df.columns:
        df = df.explode("macro_regimes")
        df = df[df["macro_regimes"].isin(filter.macro_regimes)]

    df = df.reset_index(drop=True)
    return df


def load_stats(filter: StatsFilter) -> pd.DataFrame:
    base = Path("storage/stats")
    files = sorted(base.glob("stats_*.parquet")) if base.exists() else []

    def _file_in_range(p: Path) -> bool:
        if filter.start_date is None and filter.end_date is None:
            return True
        stem = p.stem  # stats_YYYYMMDD
        try:
            part = stem.split("_")[1]
            file_date = datetime.strptime(part, "%Y%m%d").date()
        except Exception:
            return True
        if filter.start_date and file_date < filter.start_date:
            return False
        if filter.end_date and file_date > filter.end_date:
            return False
        return True

    files = [p for p in files if _file_in_range(p)]
    df = _load_parquet_files(files)
    if df.empty:
        return df

    if "timestamp_utc" in df.columns and (filter.start_date or filter.end_date):
        mask = df["timestamp_utc"].apply(
            lambda ts: _within_range(str(ts), filter.start_date, filter.end_date)
        )
        df = df[mask]

    df = _maybe_filter(df, "regime", filter.regimes)
    df = _maybe_filter(df, "feature_name", filter.feature_names)
    df = _maybe_filter(df, "run_id", filter.run_ids)
    df = df.reset_index(drop=True)
    return df


def load_policies(filter: PolicyFilter) -> pd.DataFrame:
    base = Path("storage/policies/policies.parquet")
    if not base.exists():
        return pd.DataFrame()
    df = pd.read_parquet(base)
    if df.empty:
        return df

    if "timestamp_utc" in df.columns and (filter.start_date or filter.end_date):
        mask = df["timestamp_utc"].apply(
            lambda ts: _within_range(str(ts), filter.start_date, filter.end_date)
        )
        df = df[mask]

    df = _maybe_filter(df, "policy_version", filter.policy_versions)

    if filter.engine_versions and "metadata" in df.columns:
        df = df[
            df["metadata"].apply(
                lambda m: isinstance(m, dict)
                and m.get("engine_version") in filter.engine_versions
            )
        ]

    if filter.sources and "metadata" in df.columns:
        df = df[
            df["metadata"].apply(
                lambda m: isinstance(m, dict) and m.get("source") in filter.sources
            )
        ]

    return df.reset_index(drop=True)


# --------------------------- computations ---------------------------


def _safe_std(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.std(ddof=0))


def compute_regime_performance(df_decisions: pd.DataFrame) -> pd.DataFrame:
    if df_decisions.empty:
        return pd.DataFrame()

    df = df_decisions.copy()
    if "macro_regimes" in df.columns:
        df = df.explode("macro_regimes")
    group_cols = [c for c in ["session_regime", "macro_regimes"] if c in df.columns]
    if not group_cols:
        group_cols = ["session_regime"] if "session_regime" in df.columns else []

    def _agg(group: pd.DataFrame) -> pd.Series:
        pnl = group.get("outcome_pnl")
        hit = group.get("outcome_hit")
        sharpe = 0.0
        if pnl is not None and not pnl.empty:
            mu = float(pnl.mean())
            sigma = _safe_std(pnl)
            sharpe = mu / sigma if sigma > 0 else 0.0
        hit_rate = float(hit.mean()) if hit is not None else 0.0
        return pd.Series(
            {
                "count": len(group),
                "pnl_sum": float(pnl.sum()) if pnl is not None else 0.0,
                "pnl_mean": float(pnl.mean()) if pnl is not None else 0.0,
                "pnl_variance": float(pnl.var(ddof=0)) if pnl is not None else 0.0,
                "hit_rate": hit_rate,
                "sharpe": sharpe,
            }
        )

    grouped = df.groupby(group_cols, dropna=False).apply(_agg).reset_index()
    grouped = grouped.sort_values(group_cols).reset_index(drop=True)
    return grouped


def compute_feature_drift_over_time(df_stats: pd.DataFrame) -> pd.DataFrame:
    if df_stats.empty:
        return pd.DataFrame()

    df = df_stats.copy()
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.sort_values(
        [c for c in ["feature_name", "regime", "timestamp_utc"] if c in df.columns]
    )

    records: List[dict] = []
    group_cols = [c for c in ["feature_name", "regime"] if c in df.columns]
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.reset_index(drop=True)
        stability = g.get("stability")
        variance = g.get("variance")
        for idx, row in g.iterrows():
            prev_stability = None if idx == 0 else stability.iloc[idx - 1]
            norm_change = None
            if prev_stability not in (None, 0) and stability is not None:
                try:
                    norm_change = (row.get("stability") - prev_stability) / abs(
                        prev_stability
                    )
                except Exception:
                    norm_change = None
            records.append(
                {
                    **{k: row.get(k) for k in group_cols},
                    "timestamp_utc": row.get("timestamp_utc"),
                    "variance": row.get("variance") if variance is not None else None,
                    "stability": (
                        row.get("stability") if stability is not None else None
                    ),
                    "stability_norm_change": norm_change,
                }
            )

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(
            [c for c in ["feature_name", "regime", "timestamp_utc"] if c in out.columns]
        ).reset_index(drop=True)
    return out


def compute_policy_version_performance(
    df_decisions: pd.DataFrame, df_policies: pd.DataFrame
) -> pd.DataFrame:
    if df_decisions.empty:
        return pd.DataFrame()

    df = df_decisions.copy()
    if (
        df_policies is not None
        and not df_policies.empty
        and "policy_version" in df_policies.columns
    ):
        df = df.merge(
            df_policies[["policy_version", "timestamp_utc"]].rename(
                columns={"timestamp_utc": "policy_timestamp_utc"}
            ),
            on="policy_version",
            how="left",
        )

    group_cols = ["policy_version"]

    def _agg(group: pd.DataFrame) -> pd.Series:
        pnl = group.get("outcome_pnl")
        hit = group.get("outcome_hit")
        sharpe = 0.0
        if pnl is not None and not pnl.empty:
            mu = float(pnl.mean())
            sigma = _safe_std(pnl)
            sharpe = mu / sigma if sigma > 0 else 0.0
        return pd.Series(
            {
                "count": len(group),
                "pnl_sum": float(pnl.sum()) if pnl is not None else 0.0,
                "pnl_mean": float(pnl.mean()) if pnl is not None else 0.0,
                "hit_rate": float(hit.mean()) if hit is not None else 0.0,
                "sharpe": sharpe,
                "policy_timestamp_utc_min": group.get(
                    "policy_timestamp_utc", pd.Series([None])
                ).min(),
                "policy_timestamp_utc_max": group.get(
                    "policy_timestamp_utc", pd.Series([None])
                ).max(),
            }
        )

    grouped = df.groupby(group_cols, dropna=False).apply(_agg).reset_index()
    grouped = grouped.sort_values(group_cols).reset_index(drop=True)
    return grouped
