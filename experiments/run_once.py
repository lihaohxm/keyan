"""Run a single Monte Carlo simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from sim.channel import ChannelState, generate_channels
from sim.config import SimulationConfig
from sim.effective_channel import RISPhase, compute_effective_channels, random_ris_phase
from sim.geometry import generate_geometry
from sim.matching.exhaustive import exhaustive_match
from sim.matching.norm_based import norm_based_match
from sim.matching.qoe_aware import qoe_aware_match
from sim.matching.random_match import random_match
from sim.qoe import QoEResult, compute_qoe
from sim.sinr_rate import RateResult, compute_rates


@dataclass
class RunResult:
    sum_rate: float
    avg_qoe: float
    per_user_rate: np.ndarray
    per_user_qoe: np.ndarray
    assignment: np.ndarray


def _assignment_for_algo(
    algo: str,
    channels: ChannelState,
    ris_phase: RISPhase,
    config: SimulationConfig,
    rng: np.random.Generator,
    weights: Tuple[float, float],
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
            channels, ris_phase, config, weights
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

    raise ValueError(f"Unknown algorithm: {algo}")


def _build_qoe_cost_matrix(
    channels: ChannelState,
    ris_phase: RISPhase,
    config: SimulationConfig,
    weights: Tuple[float, float],
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
            )
            cost[k, l + 1] = qoe_result.per_user[k]

    return cost


def run_once(
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

    assignment = _assignment_for_algo(algo, channels, ris_phase, config, rng, weights)
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
    )

    return RunResult(
        sum_rate=rate_result.sum_rate,
        avg_qoe=qoe_result.avg_qoe,
        per_user_rate=rate_result.rates,
        per_user_qoe=qoe_result.per_user,
        assignment=assignment,
    )
