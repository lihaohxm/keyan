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
    snr_db: Sequence[float],
    metrics_dict: Mapping[str, float],
    arrays_dict: Mapping[str, Sequence[float]],
) -> None:
    """Save a single run's results in MAT, JSON, and CSV formats.

    Args:
        run_id: Identifier for the run (used in filenames).
        snr_db: Sequence of SNR values in dB.
        metrics_dict: Scalar metrics to store in JSON.
        arrays_dict: Curve data keyed by metric name (e.g., ber, bler).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    snr_db_array = np.asarray(snr_db, dtype=float)
    mat_payload: dict[str, np.ndarray | float | int] = {
        "snr_db": snr_db_array,
    }

    for key, values in arrays_dict.items():
        mat_payload[key] = np.asarray(values, dtype=float)

    savemat(RESULTS_DIR / f"{run_id}_curves.mat", mat_payload)

    with (RESULTS_DIR / f"{run_id}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    csv_path = RESULTS_DIR / f"{run_id}_curves.csv"
    header = ["snr_db", *arrays_dict.keys()]
    rows = []
    for idx, snr_value in enumerate(snr_db_array):
        row = [snr_value]
        for key in arrays_dict.keys():
            values = np.asarray(arrays_dict[key], dtype=float)
            row.append(values[idx] if idx < len(values) else np.nan)
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
