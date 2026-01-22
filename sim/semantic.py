"""Semantic similarity proxy functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from sim.config import SemanticConfig


def score(gamma: np.ndarray, m_symbols: np.ndarray, config: SemanticConfig) -> np.ndarray:
    """Compute semantic similarity score in [0, 1]."""
    gamma = np.asarray(gamma, dtype=float)
    m_symbols = np.asarray(m_symbols, dtype=float)

    if config.method == "table" and config.table:
        table = config.table
        interpolator = RegularGridInterpolator(
            (np.asarray(table["gamma_grid"]), np.asarray(table["m_grid"])),
            np.asarray(table["values"]),
            bounds_error=False,
            fill_value=None,
        )
        points = np.stack([gamma, m_symbols], axis=-1)
        xi = interpolator(points)
        return np.clip(xi, 0.0, 1.0)

    if config.method == "ratio":
        xi = gamma / (gamma + config.c)
        xi *= 1.0 - np.exp(-config.b * m_symbols)
        return np.clip(xi, 0.0, 1.0)

    xi = (1.0 - np.exp(-config.a * gamma)) * (1.0 - np.exp(-config.b * m_symbols))
    return np.clip(xi, 0.0, 1.0)


def default_params() -> dict[str, Any]:
    return {"method": "exp", "a": 0.8, "b": 0.15, "c": 1.0}
