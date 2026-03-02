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


def _plot_curve(
    x: np.ndarray,
    curves: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(6, 4))
    for label, y in curves.items():
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel(xlabel)
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


def _latest_run_id() -> str:
    candidates = sorted(RESULTS_DIR.glob("*_metrics.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No metrics JSON files found in results/")
    return candidates[-1].stem.replace("_metrics", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot curves from saved results")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--latest", action="store_true", help="Use latest run in results/")
    args = parser.parse_args()

    if args.latest:
        run_id = _latest_run_id()
    elif args.run_id:
        run_id = args.run_id
    else:
        raise ValueError("Provide --run_id or --latest")

    curves = _load_curves(run_id)
    metrics = _load_metrics(run_id)

    x_name = metrics.get("x_name", "x")
    x = curves.get("x_axis")
    if x is None:
        raise ValueError("Expected x_axis column in curves CSV")

    sum_rate_curves = {
        name.replace("sum_rate_", ""): values
        for name, values in curves.items()
        if name.startswith("sum_rate_") and "urgent" not in name and "normal" not in name
    }
    qoe_curves = {
        name.replace("avg_qoe_", ""): values
        for name, values in curves.items()
        if name == "x_axis" or (name.startswith("avg_qoe_") and "urgent" not in name and "normal" not in name)
    }
    qoe_urgent_curves = {
        name.replace("avg_qoe_urgent_", ""): values
        for name, values in curves.items()
        if name.startswith("avg_qoe_urgent_")
    }
    qoe_normal_curves = {
        name.replace("avg_qoe_normal_", ""): values
        for name, values in curves.items()
        if name.startswith("avg_qoe_normal_")
    }

    _plot_curve(
        x,
        sum_rate_curves,
        f"Sum Rate vs {x_name}",
        x_name,
        "Sum Rate",
        f"{run_id}_sum_rate.png",
    )
    _plot_curve(
        x,
        qoe_curves,
        f"Avg QoE vs {x_name}",
        x_name,
        "Avg QoE Cost",
        f"{run_id}_avg_qoe.png",
    )
    if qoe_urgent_curves and qoe_normal_curves:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for label, y in qoe_urgent_curves.items():
            ax1.plot(x, y, marker="o", label=label)
        ax1.set_xlabel(x_name)
        ax1.set_ylabel("Avg QoE Cost (Urgent)")
        ax1.set_title("Urgent users (1–4)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        for label, y in qoe_normal_curves.items():
            ax2.plot(x, y, marker="o", label=label)
        ax2.set_xlabel(x_name)
        ax2.set_ylabel("Avg QoE Cost (Normal)")
        ax2.set_title("Normal users (5–12)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{run_id}_qoe_urgent_normal.png", dpi=200)
        plt.close()

    pareto = metrics.get("pareto")
    if pareto:
        _plot_pareto(pareto, f"{run_id}_pareto.png")
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
            _plot_pareto(pareto_curves, f"{run_id}_pareto.png")


if __name__ == "__main__":
    main()
