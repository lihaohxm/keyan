"""Compute effective channels given RIS assignments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.channel import ChannelState


@dataclass
class RISPhase:
    phases: np.ndarray


def random_ris_phase(n_ris: int, n_elements: int, rng: np.random.Generator) -> RISPhase:
    phases = rng.uniform(0.0, 2 * np.pi, size=(n_ris, n_elements))
    return RISPhase(phases=np.exp(1j * phases))


def compute_effective_channels(
    channels: ChannelState,
    assignment: np.ndarray,
    ris_phase: RISPhase,
) -> np.ndarray:
    """Compute effective channels for each user.

    Args:
        channels: ChannelState with direct and RIS channels.
        assignment: Binary array of shape (K, L + 1) with x_{k,0} for direct link.
        ris_phase: RISPhase with phase shifts.

    Returns:
        h_eff: Effective channel matrix of shape (K, Nt).
    """
    h_direct = channels.h_direct
    g_bs_ris = channels.g_bs_ris
    h_ris_ue = channels.h_ris_ue

    k_total = h_direct.shape[0]
    l_total = g_bs_ris.shape[0]

    direct_mask = assignment[:, 0][:, None]
    h_eff = h_direct * direct_mask
    for l in range(l_total):
        theta = ris_phase.phases[l]
        combined = np.zeros_like(h_direct)
        for k in range(k_total):
            if assignment[k, l + 1] <= 0:
                continue
            combined[k] = g_bs_ris[l].conj().T @ (theta * h_ris_ue[l, k])
        h_eff += combined

    return h_eff
