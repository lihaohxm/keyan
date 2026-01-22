"""Collect metrics JSON files into a summary CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")


def main() -> None:
    rows = []
    for path in RESULTS_DIR.glob("*_metrics.json"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        row = {
            "run_id": path.stem.replace("_metrics", ""),
            "x_name": payload.get("x_name"),
            "algos": ",".join(payload.get("algos", [])),
            "mc": payload.get("mc"),
            "seed": payload.get("seed"),
        }
        row.update(payload.get("metrics", {}))
        rows.append(row)

    if not rows:
        print("No metrics files found.")
        return

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "summary.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
