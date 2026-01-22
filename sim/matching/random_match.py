"""Random matching strategy."""

from __future__ import annotations

import numpy as np


def random_match(k_total: int, l_total: int, capacity: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each user to either direct link or a RIS randomly."""
    assignment = np.zeros((k_total, l_total + 1), dtype=int)
    ris_load = np.zeros(l_total, dtype=int)

    user_order = rng.permutation(k_total)
    for k in user_order:
        options = list(range(l_total + 1))
        rng.shuffle(options)
        chosen = None
        for option in options:
            if option == 0:
                chosen = 0
                break
            if ris_load[option - 1] < capacity:
                chosen = option
                ris_load[option - 1] += 1
                break
        if chosen is None:
            chosen = 0
        assignment[k, chosen] = 1

    return assignment
