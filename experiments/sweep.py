"""Parameter sweep runner for multi-RIS simulations."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.run_once import run_trial
from scripts.save_results import save_run
from sim.config import DEFAULT_POWER_DBW, DEFAULT_WEIGHT_LIST, SimulationConfig
from sim.metrics import summarize

LOGS_DIR = Path("logs")


def _parse_list(arg: str) -> List[float]:
    return [float(item.strip()) for item in arg.split(",") if item.strip()]


def _seed_for_trial(seed: int, trial: int) -> int:
    return seed * 1000 + trial


def _configure_logging(run_id: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        filemode="w",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def sweep(
    config: SimulationConfig,
    x_name: str,
    x_list: List[float],
    algos: List[str],
    mc: int,
    seed: int,
    weights: Tuple[float, float],
    run_id: str,
) -> Tuple[str, Dict[str, Dict[str, List[float]]], Dict[str, List[float]]]:
    metrics: Dict[str, Dict[str, List[float]]] = {
        algo: {
            "sum_rate_mean": [],
            "sum_rate_std": [],
            "sum_rate_ci95": [],
            "avg_qoe_mean": [],
            "avg_qoe_std": [],
            "avg_qoe_ci95": [],
        }
        for algo in algos
    }
    curves: Dict[str, List[float]] = {}

    for algo in algos:
        curves[f"sum_rate_{algo}"] = []
        curves[f"avg_qoe_{algo}"] = []

    for x_value in x_list:
        algo_sum_rates: Dict[str, List[float]] = {algo: [] for algo in algos}
        algo_avg_qoe: Dict[str, List[float]] = {algo: [] for algo in algos}

        for trial in range(mc):
            config_trial = copy.deepcopy(config)
            if x_name == "p_dbw":
                config_trial.total_power_dbw = x_value
            elif x_name == "mb":
                config_trial.geometry.ris_per_cell = int(x_value)
            elif x_name == "k0":
                config_trial.ris_capacity = int(x_value)
            else:
                raise ValueError(f"Unsupported sweep axis {x_name}")

            for algo in algos:
                result = run_trial(
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
            metrics[algo]["sum_rate_mean"].append(summary_rate.mean)
            metrics[algo]["sum_rate_std"].append(summary_rate.std)
            metrics[algo]["sum_rate_ci95"].append(summary_rate.ci95)
            metrics[algo]["avg_qoe_mean"].append(summary_qoe.mean)
            metrics[algo]["avg_qoe_std"].append(summary_qoe.std)
            metrics[algo]["avg_qoe_ci95"].append(summary_qoe.ci95)

    metrics_payload = {
        "x_name": x_name,
        "x_axis": x_list,
        "algos": algos,
        "mc": mc,
        "seed": seed,
        "config": asdict(config),
        "weights": weights,
        "metrics": metrics,
        "runtime": datetime.now().isoformat(),
    }

    save_run(run_id, x_list, metrics_payload, curves)

    return run_id, metrics, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep parameters for multi-RIS simulation")
    parser.add_argument("--x", default="p_dbw", choices=["p_dbw", "mb", "k0"], help="Sweep axis")
    parser.add_argument("--x_list", default=",".join(map(str, DEFAULT_POWER_DBW)))
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--algos", default="random,norm,qoe,exhaustive")
    parser.add_argument("--weights", default="balanced", help="Weight key or explicit w_delay,w_semantic")
    parser.add_argument("--semantic_mode", default="proxy", choices=["proxy", "table", "deepsc_stub"])
    parser.add_argument("--semantic_table", default=None, help="Path to semantic lookup table")
    args = parser.parse_args()

    config = SimulationConfig(seed=args.seed)
    config.semantic.mode = args.semantic_mode
    config.semantic.table_path = args.semantic_table
    if args.semantic_mode == "table" and not args.semantic_table:
        raise ValueError("semantic_mode=table requires --semantic_table")

    x_list = _parse_list(args.x_list)
    algos = [item.strip() for item in args.algos.split(",") if item.strip()]

    if "," in args.weights:
        w_delay, w_semantic = (float(item) for item in args.weights.split(","))
        weights = (w_delay, w_semantic)
    else:
        weights = config.qoe.weights.get(args.weights, DEFAULT_WEIGHT_LIST[1])

    run_id = f"sweep_{args.x}_{'-'.join(algos)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    _configure_logging(run_id)

    run_id, metrics, _ = sweep(
        config,
        x_name=args.x,
        x_list=x_list,
        algos=algos,
        mc=args.mc,
        seed=args.seed,
        weights=weights,
        run_id=run_id,
    )

    print(json.dumps({"run_id": run_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
