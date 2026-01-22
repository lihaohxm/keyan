"""Parameter sweep runner for multi-RIS simulations."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from experiments.run_once import run_once
from scripts.save_results import save_run
from sim.config import DEFAULT_POWER_DBW, SimulationConfig
from sim.metrics import summarize


def _parse_list(arg: str) -> List[float]:
    return [float(item.strip()) for item in arg.split(",") if item.strip()]


def _seed_for_trial(seed: int, trial: int) -> int:
    return seed * 1000 + trial


def sweep(
    config: SimulationConfig,
    x_name: str,
    x_list: List[float],
    algos: List[str],
    mc: int,
    seed: int,
    weight_key: str,
) -> Tuple[str, Dict[str, Dict[str, float]], Dict[str, List[float]]]:
    metrics: Dict[str, Dict[str, float]] = {}
    curves: Dict[str, List[float]] = {}

    for algo in algos:
        curves[f"sum_rate_{algo}"] = []
        curves[f"avg_qoe_{algo}"] = []

    weights = config.qoe.weights[weight_key]

    for x_value in x_list:
        algo_sum_rates: Dict[str, List[float]] = {algo: [] for algo in algos}
        algo_avg_qoe: Dict[str, List[float]] = {algo: [] for algo in algos}

        for trial in range(mc):
            config_trial = copy.deepcopy(config)
            if x_name == "P_dBW":
                config_trial.total_power_dbw = x_value
            elif x_name == "Mb":
                config_trial.geometry.ris_per_cell = int(x_value)
            elif x_name == "K0":
                config_trial.ris_capacity = int(x_value)
            else:
                raise ValueError(f"Unsupported sweep axis {x_name}")

            for algo in algos:
                result = run_once(
                    config_trial,
                    algo,
                    seed=_seed_for_trial(seed, trial),
                    weights=weights,
                )
                algo_sum_rates[algo].append(result.sum_rate)
                algo_avg_qoe[algo].append(result.avg_qoe)

        for algo in algos:
            summary_rate = summarize(algo_sum_rates[algo])
            summary_qoe = summarize(algo_avg_qoe[algo])
            curves[f"sum_rate_{algo}"].append(summary_rate.mean)
            curves[f"avg_qoe_{algo}"].append(summary_qoe.mean)
            metrics[f"{algo}_sum_rate_mean"] = summary_rate.mean
            metrics[f"{algo}_sum_rate_std"] = summary_rate.std
            metrics[f"{algo}_sum_rate_ci95"] = summary_rate.ci95
            metrics[f"{algo}_avg_qoe_mean"] = summary_qoe.mean
            metrics[f"{algo}_avg_qoe_std"] = summary_qoe.std
            metrics[f"{algo}_avg_qoe_ci95"] = summary_qoe.ci95

    run_id = f"sweep_{x_name}_{'-'.join(algos)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    metrics_payload = {
        "x_name": x_name,
        "x_list": x_list,
        "algos": algos,
        "mc": mc,
        "seed": seed,
        "config": asdict(config),
        "weight_key": weight_key,
        "metrics": metrics,
    }

    save_run(run_id, x_list, metrics_payload, curves)

    return run_id, metrics_payload, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep parameters for multi-RIS simulation")
    parser.add_argument("--x", default="P_dBW", choices=["P_dBW", "Mb", "K0"], help="Sweep axis")
    parser.add_argument("--x_list", default=",".join(map(str, DEFAULT_POWER_DBW)))
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--algos", default="random,norm,qoe,exhaustive")
    parser.add_argument("--weights", default="balanced", help="Weight key in QoE config")
    args = parser.parse_args()

    config = SimulationConfig(seed=args.seed)
    x_list = _parse_list(args.x_list)
    algos = [item.strip() for item in args.algos.split(",") if item.strip()]

    run_id, metrics_payload, _ = sweep(
        config,
        x_name=args.x,
        x_list=x_list,
        algos=algos,
        mc=args.mc,
        seed=args.seed,
        weight_key=args.weights,
    )

    print(json.dumps({"run_id": run_id, "metrics": metrics_payload}, indent=2))


if __name__ == "__main__":
    main()
