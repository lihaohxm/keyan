"""QoE cost computation for delay and semantic distortion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.config import QoEConfig, SemanticConfig
from sim.semantic import score as semantic_score


@dataclass
class QoEResult:
    per_user: np.ndarray
    avg_qoe: float
    sum_qoe: float
    delay_cost: np.ndarray
    semantic_cost: np.ndarray
    distortion: np.ndarray
    delay_ms: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_qoe(
    sinr: np.ndarray,
    rates: np.ndarray,
    payload_symbols: np.ndarray,
    config: QoEConfig,
    semantic_config: SemanticConfig,
    weights: tuple[float, float],
) -> QoEResult:
    """Compute QoE costs for each user and aggregate metrics."""
    payload_symbols = np.asarray(payload_symbols, dtype=float)
    xi = semantic_score(sinr, payload_symbols, semantic_config)
    distortion = 1.0 - xi

    payload = config.payload_ratio * payload_symbols
    rates = np.asarray(rates, dtype=float)
    safe_rates = np.where(rates > 1e-9, rates, 1e-9)
    delay_sec = payload / safe_rates
    delay_ms = delay_sec * 1e3

    delay_thresholds = np.asarray(config.delay_thresholds_ms)
    hard_mask = np.zeros_like(delay_ms, dtype=bool)
    hard_mask[: int(len(delay_ms) * config.hard_ratio)] = True
    rng = np.random.default_rng(0)
    rng.shuffle(hard_mask)

    delay_target = np.where(hard_mask, delay_thresholds[0], delay_thresholds[1])
    delay_cost = _sigmoid((delay_ms - delay_target) / config.beta_delay)
    delay_cost += (delay_ms > delay_target) * config.penalty_delay

    semantic_threshold = config.semantic_threshold
    semantic_cost = _sigmoid((distortion - semantic_threshold) / config.beta_semantic)
    semantic_cost += (distortion > semantic_threshold) * config.penalty_semantic

    w_delay, w_semantic = weights
    per_user = w_delay * delay_cost + w_semantic * semantic_cost

    return QoEResult(
        per_user=per_user,
        avg_qoe=float(np.mean(per_user)),
        sum_qoe=float(np.sum(per_user)),
        delay_cost=delay_cost,
        semantic_cost=semantic_cost,
        distortion=distortion,
        delay_ms=delay_ms,
    )
