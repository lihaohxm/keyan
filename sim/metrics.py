"""Statistical aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MetricSummary:
    mean: float
    std: float
    ci95: float


def summarize(values: Sequence[float]) -> MetricSummary:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return MetricSummary(mean=mean, std=std, ci95=ci95)
