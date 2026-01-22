"""QoE-aware matching using min-cost assignment with capacities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def qoe_aware_match(cost_matrix: np.ndarray, capacity: int) -> np.ndarray:
    """Solve min-cost assignment with RIS capacity via slot expansion.

    Args:
        cost_matrix: Array of shape (K, L + 1) with QoE costs for each option.
        capacity: Max users per RIS.

    Returns:
        assignment: Binary array of shape (K, L + 1).
    """
    k_total, l_plus = cost_matrix.shape
    l_total = l_plus - 1

    slot_options = []
    for l in range(1, l_plus):
        for _ in range(capacity):
            slot_options.append(l)
    for _ in range(k_total):
        slot_options.append(0)

    slot_options = np.asarray(slot_options, dtype=int)
    expanded_cost = cost_matrix[:, slot_options]

    row_ind, col_ind = linear_sum_assignment(expanded_cost)
    assignment = np.zeros((k_total, l_plus), dtype=int)
    for k, slot in zip(row_ind, col_ind):
        assignment[k, slot_options[slot]] = 1

    return assignment
