"""Pareto sweep across QoE weights."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.run_once import run_trial
from scripts.save_results import save_run
from sim.config import DEFAULT_WEIGHT_LIST, SimulationConfig
from sim.metrics import summarize

LOGS_DIR = Path("logs")


def _parse_weights(arg: str) -> List[Tuple[float, float]]:
    weights = []
    for item in arg.split(";"):
        if not item.strip():
            continue
        w_delay, w_semantic = item.split(",")
        weights.append((float(w_delay), float(w_semantic)))
    return weights


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


def pareto(
    config: SimulationConfig,
    weight_list: List[Tuple[float, float]],
    algos: List[str],
    mc: int,
    seed: int,
    run_id: str,
) -> Tuple[str, Dict[str, Dict[str, List[float]]]]:
    pareto_points: Dict[str, Dict[str, List[float]]] = {
        algo: {"sum_rate": [], "avg_qoe": []} for algo in algos
    }

    for weights in weight_list:
        for algo in algos:
            sum_rates = []
            avg_qoes = []
            for trial in range(mc):
                result = run_trial(
                    config,
                    algo,
                    seed=_seed_for_trial(seed, trial),
                    weights=weights,
                )
                sum_rates.append(result.sum_rate)
                avg_qoes.append(result.avg_qoe)

            pareto_points[algo]["sum_rate"].append(summarize(sum_rates).mean)
            pareto_points[algo]["avg_qoe"].append(summarize(avg_qoes).mean)

    metrics_payload = {
        "weight_list": weight_list,
        "algos": algos,
        "mc": mc,
        "seed": seed,
        "config": asdict(config),
        "pareto": pareto_points,
        "x_name": "weights",
        "x_axis": list(range(len(weight_list))),
        "runtime": datetime.now().isoformat(),
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
    parser.add_argument("--semantic_mode", default="proxy", choices=["proxy", "table", "deepsc_stub"])
    parser.add_argument("--semantic_table", default=None, help="Path to semantic lookup table")
    args = parser.parse_args()

    config = SimulationConfig(seed=args.seed)
    config.semantic.mode = args.semantic_mode
    config.semantic.table_path = args.semantic_table
    if args.semantic_mode == "table" and not args.semantic_table:
        raise ValueError("semantic_mode=table requires --semantic_table")

    if args.weights:
        weight_list = _parse_weights(args.weights)
    else:
        weight_list = DEFAULT_WEIGHT_LIST

    algos = [item.strip() for item in args.algos.split(",") if item.strip()]

    run_id = f"pareto_{'-'.join(algos)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    _configure_logging(run_id)

    run_id, pareto_points = pareto(
        config=config,
        weight_list=weight_list,
        algos=algos,
        mc=args.mc,
        seed=args.seed,
        run_id=run_id,
    )

    print(json.dumps({"run_id": run_id, "pareto": pareto_points}, indent=2))


if __name__ == "__main__":
    main()
