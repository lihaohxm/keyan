"""Exhaustive search matching (small-scale upper bound)."""

from __future__ import annotations

import itertools
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def exhaustive_match(
    k_total: int,
    l_total: int,
    capacity: int,
    objective: Callable[[np.ndarray], float],
    max_users: int = 6,
    max_ris: int = 4,
) -> np.ndarray | None:
    """Enumerate all assignments and return the best one if small enough."""
    if k_total > max_users or l_total > max_ris:
        logger.info(
            "Exhaustive matching skipped (K=%s, L=%s exceeds limits).",
            k_total,
            l_total,
        )
        return None

    best_score = -np.inf
    best_assignment: np.ndarray | None = None

    options = list(range(l_total + 1))
    for combo in itertools.product(options, repeat=k_total):
        counts = np.zeros(l_total, dtype=int)
        valid = True
        for choice in combo:
            if choice == 0:
                continue
            counts[choice - 1] += 1
            if counts[choice - 1] > capacity:
                valid = False
                break
        if not valid:
            continue

        assignment = np.zeros((k_total, l_total + 1), dtype=int)
        for k, choice in enumerate(combo):
            assignment[k, choice] = 1

        score = objective(assignment)
        if score > best_score:
            best_score = score
            best_assignment = assignment

    return best_assignment
