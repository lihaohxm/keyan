"""Utilities for saving experimental results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.io import savemat

RESULTS_DIR = Path("results")


def save_run(
    run_id: str,
    x_axis: Sequence[float],
    metrics_dict: Mapping[str, object],
    arrays_dict: Mapping[str, Sequence[float]],
) -> None:
    """Save a single run's results in MAT, JSON, and CSV formats.

    Args:
        run_id: Identifier for the run (used in filenames).
        x_axis: Sequence of x-axis values.
        metrics_dict: Scalar metrics to store in JSON.
        arrays_dict: Curve data keyed by metric name.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_axis_array = np.asarray(x_axis, dtype=float)
    mat_payload: dict[str, np.ndarray | float | int] = {
        "x_axis": x_axis_array,
    }

    for key, values in arrays_dict.items():
        mat_payload[key] = np.asarray(values, dtype=float)

    savemat(RESULTS_DIR / f"{run_id}_curves.mat", mat_payload)

    with (RESULTS_DIR / f"{run_id}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    csv_path = RESULTS_DIR / f"{run_id}_curves.csv"
    header = ["x_axis", *arrays_dict.keys()]
    rows = []
    for idx, x_value in enumerate(x_axis_array):
        row = [x_value]
        for key in arrays_dict.keys():
            values = np.asarray(arrays_dict[key], dtype=float)
            row.append(values[idx] if idx < len(values) else np.nan)
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
