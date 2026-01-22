"""QoE cost computation for delay and semantic distortion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.config import QoEConfig, SemanticConfig
from sim.semantic import get_xi


@dataclass
class QoEResult:
    per_user: np.ndarray
    avg_qoe: float
    sum_qoe: float
    delay_cost: np.ndarray
    semantic_cost: np.ndarray
    distortion: np.ndarray
    delay_ms: np.ndarray
    deadline_success_rate: float
    semantic_success_rate: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_mask(k_total: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    count = int(round(k_total * ratio))
    mask = np.zeros(k_total, dtype=bool)
    if count > 0:
        mask[:count] = True
    rng.shuffle(mask)
    return mask


def compute_qoe(
    sinr: np.ndarray,
    rates: np.ndarray,
    payload_symbols: np.ndarray,
    config: QoEConfig,
    semantic_config: SemanticConfig,
    weights: tuple[float, float],
    rng: np.random.Generator,
    hard_mask: np.ndarray | None = None,
    delay_mask: np.ndarray | None = None,
) -> QoEResult:
    """Compute QoE costs for each user and aggregate metrics."""
    payload_symbols = np.asarray(payload_symbols, dtype=float)
    xi = get_xi(
        sinr,
        payload_symbols,
        semantic_config.mode,
        table_path=semantic_config.table_path,
        params={"a": semantic_config.a, "b": semantic_config.b},
    )
    distortion = 1.0 - xi

    payload = config.payload_ratio * payload_symbols
    rates = np.asarray(rates, dtype=float)
    safe_rates = np.where(rates > 1e-9, rates, 1e-9)
    delay_sec = payload / safe_rates
    delay_ms = delay_sec * 1e3

    k_total = delay_ms.size
    if hard_mask is None:
        hard_mask = _sample_mask(k_total, config.hard_ratio, rng)
    if delay_mask is None:
        delay_mask = _sample_mask(k_total, config.delay_ratio, rng)

    delay_thresholds = np.asarray(config.delay_thresholds_ms, dtype=float)
    delay_target = np.where(delay_mask, delay_thresholds[0], delay_thresholds[1])
    delay_cost = _sigmoid((delay_ms - delay_target) / config.beta_delay)
    delay_cost += (delay_ms > delay_target) * hard_mask * config.penalty_delay

    semantic_target = config.distortion_max
    semantic_cost = _sigmoid((distortion - semantic_target) / config.beta_semantic)
    semantic_cost += (distortion > semantic_target) * hard_mask * config.penalty_semantic

    w_delay, w_semantic = weights
    per_user = w_delay * delay_cost + w_semantic * semantic_cost

    deadline_success_rate = float(np.mean(delay_ms <= delay_target))
    semantic_success_rate = float(np.mean(distortion <= semantic_target))

    return QoEResult(
        per_user=per_user,
        avg_qoe=float(np.mean(per_user)),
        sum_qoe=float(np.sum(per_user)),
        delay_cost=delay_cost,
        semantic_cost=semantic_cost,
        distortion=distortion,
        delay_ms=delay_ms,
        deadline_success_rate=deadline_success_rate,
        semantic_success_rate=semantic_success_rate,
    )
