"""Run a single Monte Carlo simulation."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from scripts.save_results import save_run
from sim.channel import ChannelState, generate_channels
from sim.config import DEFAULT_WEIGHT_LIST, SimulationConfig
from sim.effective_channel import RISPhase, compute_effective_channels, random_ris_phase
from sim.geometry import generate_geometry
from sim.matching.exhaustive import exhaustive_match
from sim.matching.ga_placeholder import ga_match
from sim.matching.norm_based import norm_based_match
from sim.matching.qoe_aware import qoe_aware_match
from sim.matching.random_match import random_match
from sim.metrics import summarize
from sim.qoe import QoEResult, compute_qoe
from sim.sinr_rate import RateResult, compute_rates

LOGS_DIR = Path("logs")


@dataclass
class RunResult:
    sum_rate: float
    avg_qoe: float
    per_user_rate: np.ndarray
    per_user_qoe: np.ndarray
    assignment: np.ndarray
    qoe_result: QoEResult
    rate_result: RateResult


def _configure_logging(run_id: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        filemode="w",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _assignment_for_algo(
    algo: str,
    channels: ChannelState,
    ris_phase: RISPhase,
    config: SimulationConfig,
    rng: np.random.Generator,
    weights: Tuple[float, float],
    hard_mask: np.ndarray,
    delay_mask: np.ndarray,
) -> np.ndarray:
    k_total = channels.h_direct.shape[0]
    l_total = channels.g_bs_ris.shape[0]

    if algo == "random":
        return random_match(k_total, l_total, config.ris_capacity, rng)
    if algo == "norm":
        return norm_based_match(
            channels.g_bs_ris,
            channels.h_ris_ue,
            ris_phase.phases,
            config.ris_capacity,
        )
    if algo == "qoe":
        cost_matrix = _build_qoe_cost_matrix(
            channels, ris_phase, config, weights, rng, hard_mask, delay_mask
        )
        return qoe_aware_match(cost_matrix, config.ris_capacity)
    if algo == "exhaustive":
        def objective(assignment: np.ndarray) -> float:
            h_eff = compute_effective_channels(channels, assignment, ris_phase)
            rate_result = compute_rates(
                h_eff,
                total_power=10 ** (config.total_power_dbw / 10.0),
                noise_power=channels.noise_power,
                bandwidth=config.bandwidth_hz,
            )
            return rate_result.sum_rate

        assignment = exhaustive_match(
            k_total,
            l_total,
            config.ris_capacity,
            objective,
        )
        if assignment is None:
            return random_match(k_total, l_total, config.ris_capacity, rng)
        return assignment
    if algo == "ga_placeholder":
        return ga_match()

    raise ValueError(f"Unknown algorithm: {algo}")


def _build_qoe_cost_matrix(
    channels: ChannelState,
    ris_phase: RISPhase,
    config: SimulationConfig,
    weights: Tuple[float, float],
    rng: np.random.Generator,
    hard_mask: np.ndarray,
    delay_mask: np.ndarray,
) -> np.ndarray:
    k_total = channels.h_direct.shape[0]
    l_total = channels.g_bs_ris.shape[0]
    cost = np.zeros((k_total, l_total + 1))

    for k in range(k_total):
        assignment = np.zeros((k_total, l_total + 1), dtype=int)
        assignment[k, 0] = 1
        h_eff = compute_effective_channels(channels, assignment, ris_phase)
        rate_result = compute_rates(
            h_eff,
            total_power=10 ** (config.total_power_dbw / 10.0),
            noise_power=channels.noise_power,
            bandwidth=config.bandwidth_hz,
        )
        qoe_result = compute_qoe(
            rate_result.sinr,
            rate_result.rates,
            np.full(k_total, config.payload_symbols),
            config.qoe,
            config.semantic,
            weights,
            rng,
            hard_mask=hard_mask,
            delay_mask=delay_mask,
        )
        cost[k, 0] = qoe_result.per_user[k]

        for l in range(l_total):
            assignment = np.zeros((k_total, l_total + 1), dtype=int)
            assignment[k, l + 1] = 1
            h_eff = compute_effective_channels(channels, assignment, ris_phase)
            rate_result = compute_rates(
                h_eff,
                total_power=10 ** (config.total_power_dbw / 10.0),
                noise_power=channels.noise_power,
                bandwidth=config.bandwidth_hz,
            )
            qoe_result = compute_qoe(
                rate_result.sinr,
                rate_result.rates,
                np.full(k_total, config.payload_symbols),
                config.qoe,
                config.semantic,
                weights,
                rng,
                hard_mask=hard_mask,
                delay_mask=delay_mask,
            )
            cost[k, l + 1] = qoe_result.per_user[k]

    return cost


def run_trial(
    config: SimulationConfig,
    algo: str,
    seed: int,
    weights: Tuple[float, float],
) -> RunResult:
    rng = np.random.default_rng(seed)
    geometry = generate_geometry(config.geometry, rng)
    channels = generate_channels(geometry, config.channel, rng)
    ris_phase = random_ris_phase(
        channels.g_bs_ris.shape[0], config.channel.ris_elements, rng
    )

    k_total = channels.h_direct.shape[0]
    hard_mask = np.zeros(k_total, dtype=bool)
    hard_mask[: int(round(k_total * config.qoe.hard_ratio))] = True
    rng.shuffle(hard_mask)

    delay_mask = np.zeros(k_total, dtype=bool)
    delay_mask[: int(round(k_total * config.qoe.delay_ratio))] = True
    rng.shuffle(delay_mask)

    assignment = _assignment_for_algo(
        algo, channels, ris_phase, config, rng, weights, hard_mask, delay_mask
    )
    h_eff = compute_effective_channels(channels, assignment, ris_phase)

    rate_result = compute_rates(
        h_eff,
        total_power=10 ** (config.total_power_dbw / 10.0),
        noise_power=channels.noise_power,
        bandwidth=config.bandwidth_hz,
    )

    qoe_result = compute_qoe(
        rate_result.sinr,
        rate_result.rates,
        np.full(rate_result.rates.shape[0], config.payload_symbols),
        config.qoe,
        config.semantic,
        weights,
        rng,
        hard_mask=hard_mask,
        delay_mask=delay_mask,
    )

    return RunResult(
        sum_rate=rate_result.sum_rate,
        avg_qoe=qoe_result.avg_qoe,
        per_user_rate=rate_result.rates,
        per_user_qoe=qoe_result.per_user,
        assignment=assignment,
        qoe_result=qoe_result,
        rate_result=rate_result,
    )


def _seed_for_trial(seed: int, trial: int) -> int:
    return seed * 1000 + trial


def run_once(
    config: SimulationConfig,
    algos: list[str],
    mc: int,
    seed: int,
    weights: Tuple[float, float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, list[float]]]:
    metrics: Dict[str, Dict[str, float]] = {algo: {} for algo in algos}
    curves: Dict[str, list[float]] = {}

    for algo in algos:
        curves[f"sum_rate_{algo}"] = []
        curves[f"avg_qoe_{algo}"] = []

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

        summary_rate = summarize(sum_rates)
        summary_qoe = summarize(avg_qoes)
        curves[f"sum_rate_{algo}"].append(summary_rate.mean)
        curves[f"avg_qoe_{algo}"].append(summary_qoe.mean)
        metrics[algo] = {
            "sum_rate_mean": summary_rate.mean,
            "sum_rate_std": summary_rate.std,
            "sum_rate_ci95": summary_rate.ci95,
            "avg_qoe_mean": summary_qoe.mean,
            "avg_qoe_std": summary_qoe.std,
            "avg_qoe_ci95": summary_qoe.ci95,
        }

    return metrics, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Monte Carlo configuration")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mc", type=int, default=20)
    parser.add_argument("--p_dbw", type=float, default=-5.0)
    parser.add_argument("--algos", default="random,norm,qoe,exhaustive")
    parser.add_argument("--semantic_mode", default="proxy", choices=["proxy", "table", "deepsc_stub"])
    parser.add_argument("--semantic_table", default=None, help="Path to semantic lookup table")
    parser.add_argument("--weights", default=None, help="Override weights as w_delay,w_semantic")
    args = parser.parse_args()

    run_id = f"run_once_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    _configure_logging(run_id)

    config = SimulationConfig(seed=args.seed)
    config.total_power_dbw = args.p_dbw
    config.semantic.mode = args.semantic_mode
    config.semantic.table_path = args.semantic_table
    if args.semantic_mode == "table" and not args.semantic_table:
        raise ValueError("semantic_mode=table requires --semantic_table")

    if args.weights:
        w_delay, w_semantic = (float(item) for item in args.weights.split(","))
        weights = (w_delay, w_semantic)
    else:
        weights = DEFAULT_WEIGHT_LIST[1]

    algos = [item.strip() for item in args.algos.split(",") if item.strip()]

    metrics, curves = run_once(
        config=config,
        algos=algos,
        mc=args.mc,
        seed=args.seed,
        weights=weights,
    )

    payload = {
        "x_name": "p_dbw",
        "x_axis": [args.p_dbw],
        "algos": algos,
        "mc": args.mc,
        "seed": args.seed,
        "config": asdict(config),
        "weights": weights,
        "metrics": metrics,
        "runtime": datetime.now().isoformat(),
    }

    save_run(run_id, [args.p_dbw], payload, curves)

    output = {
        "run_id": run_id,
        "results": payload,
        "files": {
            "metrics": f"results/{run_id}_metrics.json",
            "curves": f"results/{run_id}_curves.csv",
            "mat": f"results/{run_id}_curves.mat",
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
