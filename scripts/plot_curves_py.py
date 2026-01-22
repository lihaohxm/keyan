"""Plot curves from saved results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def _load_curves(run_id: str) -> Dict[str, np.ndarray]:
    csv_path = RESULTS_DIR / f"{run_id}_curves.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


def _load_metrics(run_id: str) -> dict:
    json_path = RESULTS_DIR / f"{run_id}_metrics.json"
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_curve(x: np.ndarray, curves: Dict[str, np.ndarray], title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    for label, y in curves.items():
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def _plot_pareto(pareto: Dict[str, Dict[str, List[float]]], filename: str) -> None:
    plt.figure(figsize=(6, 4))
    for algo, values in pareto.items():
        x = values["avg_qoe"]
        y = values["sum_rate"]
        plt.plot(x, y, marker="o", label=algo)
    plt.xlabel("Average QoE Cost")
    plt.ylabel("Sum Rate")
    plt.title("Pareto Tradeoff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot curves from saved results")
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    curves = _load_curves(args.run_id)
    metrics = _load_metrics(args.run_id)

    x_name = metrics.get("x_name", "x")
    x = curves.get("snr_db")
    if x is None:
        raise ValueError("Expected snr_db column in curves CSV")

    sum_rate_curves = {
        name.replace("sum_rate_", ""): values
        for name, values in curves.items()
        if name.startswith("sum_rate_")
    }
    qoe_curves = {
        name.replace("avg_qoe_", ""): values
        for name, values in curves.items()
        if name.startswith("avg_qoe_")
    }

    _plot_curve(x, sum_rate_curves, f"Sum Rate vs {x_name}", "Sum Rate", f"{args.run_id}_sum_rate.png")
    _plot_curve(x, qoe_curves, f"Avg QoE vs {x_name}", "Avg QoE", f"{args.run_id}_avg_qoe.png")

    pareto = metrics.get("pareto")
    if pareto:
        _plot_pareto(pareto, f"{args.run_id}_pareto.png")
    else:
        pareto_curves = {}
        for name, values in curves.items():
            if name.startswith("pareto_sum_rate_"):
                algo = name.replace("pareto_sum_rate_", "")
                pareto_curves.setdefault(algo, {})["sum_rate"] = values.tolist()
            if name.startswith("pareto_avg_qoe_"):
                algo = name.replace("pareto_avg_qoe_", "")
                pareto_curves.setdefault(algo, {})["avg_qoe"] = values.tolist()
        if pareto_curves:
            _plot_pareto(pareto_curves, f"{args.run_id}_pareto.png")


if __name__ == "__main__":
    main()
