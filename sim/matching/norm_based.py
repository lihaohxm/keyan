"""Norm-based greedy matching strategy."""

from __future__ import annotations

import numpy as np


def norm_based_match(
    g_bs_ris: np.ndarray,
    h_ris_ue: np.ndarray,
    ris_phase: np.ndarray,
    capacity: int,
) -> np.ndarray:
    """Assign users to RISs based on channel norm scores."""
    l_total, k_total, _ = h_ris_ue.shape
    assignment = np.zeros((k_total, l_total + 1), dtype=int)
    ris_load = np.zeros(l_total, dtype=int)

    scores = np.zeros((k_total, l_total))
    for l in range(l_total):
        theta = ris_phase[l]
        for k in range(k_total):
            combined = g_bs_ris[l].conj().T @ (theta * h_ris_ue[l, k])
            scores[k, l] = np.linalg.norm(combined)

    for k in range(k_total):
        ordered = np.argsort(scores[k])[::-1]
        chosen = 0
        for l in ordered:
            if ris_load[l] < capacity:
                chosen = l + 1
                ris_load[l] += 1
                break
        assignment[k, chosen] = 1

    return assignment
