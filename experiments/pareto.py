"""Pareto sweep across QoE weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from experiments.run_once import run_once
from scripts.save_results import save_run
from sim.config import SimulationConfig
from sim.metrics import summarize


def _parse_weights(arg: str) -> List[Tuple[float, float]]:
    weights = []
    for item in arg.split(";"):
        if not item.strip():
            continue
        w_delay, w_semantic = item.split(",")
        weights.append((float(w_delay), float(w_semantic)))
    return weights


def pareto(
    config: SimulationConfig,
    weight_list: List[Tuple[float, float]],
    algos: List[str],
    mc: int,
    seed: int,
) -> Tuple[str, Dict[str, Dict[str, List[float]]]]:
    pareto_points: Dict[str, Dict[str, List[float]]] = {algo: {"sum_rate": [], "avg_qoe": []} for algo in algos}

    for weights in weight_list:
        for algo in algos:
            sum_rates = []
            avg_qoes = []
            for trial in range(mc):
                result = run_once(
                    config,
                    algo,
                    seed=seed * 1000 + trial,
                    weights=weights,
                )
                sum_rates.append(result.sum_rate)
                avg_qoes.append(result.avg_qoe)

            pareto_points[algo]["sum_rate"].append(summarize(sum_rates).mean)
            pareto_points[algo]["avg_qoe"].append(summarize(avg_qoes).mean)

    run_id = f"pareto_{'-'.join(algos)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    metrics_payload = {
        "weight_list": weight_list,
        "algos": algos,
        "mc": mc,
        "seed": seed,
        "config": asdict(config),
        "pareto": pareto_points,
        "x_name": "weights",
    }

    arrays_dict = {}
    for algo in algos:
        arrays_dict[f"pareto_sum_rate_{algo}"] = pareto_points[algo]["sum_rate"]
        arrays_dict[f"pareto_avg_qoe_{algo}"] = pareto_points[algo]["avg_qoe"]

    save_run(run_id, list(range(len(weight_list))), metrics_payload, arrays_dict)

    return run_id, pareto_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto sweep for QoE weights")
    parser.add_argument("--weights", default="0.8,0.2;0.5,0.5;0.2,0.8")
    parser.add_argument("--algos", default="random,norm,qoe")
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = SimulationConfig(seed=args.seed)
    weight_list = _parse_weights(args.weights)
    algos = [item.strip() for item in args.algos.split(",") if item.strip()]

    run_id, pareto_points = pareto(
        config=config,
        weight_list=weight_list,
        algos=algos,
        mc=args.mc,
        seed=args.seed,
    )

    print(json.dumps({"run_id": run_id, "pareto": pareto_points}, indent=2))


if __name__ == "__main__":
    main()
