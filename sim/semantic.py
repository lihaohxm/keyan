"""Semantic similarity mappings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from sim.config import SemanticConfig


@dataclass(frozen=True)
class SemanticTable:
    gamma_grid: np.ndarray
    m_grid: np.ndarray
    values: np.ndarray
    axis_in_db: bool


def _proxy_xi(gamma: np.ndarray, m_symbols: np.ndarray, a: float, b: float) -> np.ndarray:
    xi = (1.0 - np.exp(-a * gamma)) * (1.0 - np.exp(-b * m_symbols))
    return np.clip(xi, 0.0, 1.0)


def _as_numeric(values: np.ndarray | list[Any]) -> np.ndarray:
    return np.asarray(pd.to_numeric(np.asarray(values).ravel(), errors="coerce"), dtype=float)


def _load_csv_table(path: Path) -> SemanticTable:
    df = pd.read_csv(path)
    columns = [col.strip().lower() for col in df.columns]

    if {"snr_db", "m", "xi"}.issubset(columns) or {"gamma", "m", "xi"}.issubset(columns):
        gamma_key = "snr_db" if "snr_db" in columns else "gamma"
        axis_in_db = gamma_key == "snr_db"
        df.columns = columns
        df["m"] = pd.to_numeric(df["m"], errors="coerce")
        df[gamma_key] = pd.to_numeric(df[gamma_key], errors="coerce")
        df["xi"] = pd.to_numeric(df["xi"], errors="coerce")
        pivot = df.pivot_table(index=gamma_key, columns="m", values="xi", aggfunc="mean")
        gamma_grid = pivot.index.to_numpy(dtype=float)
        m_grid = pivot.columns.to_numpy(dtype=float)
        values = pivot.to_numpy(dtype=float)
        return SemanticTable(gamma_grid=gamma_grid, m_grid=m_grid, values=values, axis_in_db=axis_in_db)

    if len(df.columns) < 2:
        raise ValueError("CSV table must have at least two columns")

    axis_name = columns[0]
    axis_in_db = axis_name == "snr_db"
    if axis_name not in {"snr_db", "gamma"}:
        raise ValueError("First column must be snr_db or gamma for grid CSV format")

    gamma_grid = _as_numeric(df.iloc[:, 0].to_numpy())
    m_grid = _as_numeric(df.columns[1:])
    values = df.iloc[:, 1:].to_numpy(dtype=float)
    return SemanticTable(gamma_grid=gamma_grid, m_grid=m_grid, values=values, axis_in_db=axis_in_db)


def _load_npz_table(path: Path) -> SemanticTable:
    payload = np.load(path, allow_pickle=True)
    keys = {key.lower(): key for key in payload.keys()}
    gamma_key = keys.get("gamma_grid") or keys.get("snr_db_grid") or keys.get("snr_db")
    m_key = keys.get("m_grid") or keys.get("m")
    values_key = keys.get("values") or keys.get("xi")

    if gamma_key is None or m_key is None or values_key is None:
        raise ValueError("NPZ table requires gamma_grid/snr_db_grid, m_grid, values")

    gamma_grid = np.asarray(payload[gamma_key], dtype=float)
    m_grid = np.asarray(payload[m_key], dtype=float)
    values = np.asarray(payload[values_key], dtype=float)
    axis_in_db = "snr" in gamma_key.lower()
    return SemanticTable(gamma_grid=gamma_grid, m_grid=m_grid, values=values, axis_in_db=axis_in_db)


def _load_mat_table(path: Path) -> SemanticTable:
    payload = loadmat(path)
    keys = {key.lower(): key for key in payload.keys()}
    gamma_key = keys.get("gamma_grid") or keys.get("snr_db_grid") or keys.get("snr_db")
    m_key = keys.get("m_grid") or keys.get("m")
    values_key = keys.get("values") or keys.get("xi")

    if gamma_key is None or m_key is None or values_key is None:
        raise ValueError("MAT table requires gamma_grid/snr_db_grid, m_grid, values")

    gamma_grid = np.squeeze(np.asarray(payload[gamma_key], dtype=float))
    m_grid = np.squeeze(np.asarray(payload[m_key], dtype=float))
    values = np.asarray(payload[values_key], dtype=float)
    axis_in_db = "snr" in gamma_key.lower()
    return SemanticTable(gamma_grid=gamma_grid, m_grid=m_grid, values=values, axis_in_db=axis_in_db)


@lru_cache(maxsize=8)
def load_semantic_table(path: str) -> SemanticTable:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Semantic table not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_table(table_path)
    if suffix == ".npz":
        return _load_npz_table(table_path)
    if suffix == ".mat":
        return _load_mat_table(table_path)
    raise ValueError("Unsupported table format. Use CSV, NPZ, or MAT.")


@lru_cache(maxsize=8)
def _get_interpolator(path: str) -> Tuple[SemanticTable, RegularGridInterpolator]:
    table = load_semantic_table(path)
    interpolator = RegularGridInterpolator(
        (table.gamma_grid, table.m_grid),
        table.values,
        bounds_error=False,
        fill_value=None,
    )
    return table, interpolator


def get_xi(
    gamma: np.ndarray,
    m_symbols: np.ndarray,
    mode: str,
    table_path: str | None = None,
    params: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Map SINR/SNR to semantic similarity score.

    Args:
        gamma: Linear SINR values.
        m_symbols: Payload symbol counts.
        mode: "proxy", "table", or "deepsc_stub".
        table_path: Optional path to a semantic lookup table.
        params: Optional proxy parameters (a, b).
    """
    gamma = np.asarray(gamma, dtype=float)
    m_symbols = np.asarray(m_symbols, dtype=float)
    gamma, m_symbols = np.broadcast_arrays(gamma, m_symbols)
    params = params or {}

    if mode == "table":
        if not table_path:
            raise ValueError("table_path is required for table mode")
        table, interpolator = _get_interpolator(table_path)
        gamma_axis = gamma
        if table.axis_in_db:
            gamma_axis = 10.0 * np.log10(np.maximum(gamma, 1e-12))
        points = np.stack([gamma_axis, m_symbols], axis=-1)
        xi = interpolator(points)
        return np.clip(xi, 0.0, 1.0)

    if mode == "deepsc_stub":
        warnings.warn(
            "deepsc_stub mode is a placeholder; falling back to proxy mapping.",
            RuntimeWarning,
        )
        return _proxy_xi(
            gamma,
            m_symbols,
            params.get("a", 0.8),
            params.get("b", 0.15),
        )

    return _proxy_xi(
        gamma,
        m_symbols,
        params.get("a", 0.8),
        params.get("b", 0.15),
    )


def score(gamma: np.ndarray, m_symbols: np.ndarray, config: SemanticConfig) -> np.ndarray:
    """Backward-compatible wrapper for legacy callers."""
    return get_xi(
        gamma,
        m_symbols,
        config.mode,
        table_path=config.table_path,
        params={"a": config.a, "b": config.b},
    )


def default_params() -> dict[str, Any]:
    return {"mode": "proxy", "a": 0.8, "b": 0.15}
