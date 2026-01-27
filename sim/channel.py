"""Channel generation with path loss and Rayleigh fading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.config import ChannelConfig
from sim.geometry import Geometry


@dataclass
class ChannelState:
    h_direct: np.ndarray
    g_bs_ris: np.ndarray
    h_ris_ue: np.ndarray
    noise_power: float


def _pathloss(distance: np.ndarray, exponent: float) -> np.ndarray:
    distance = np.maximum(distance, 1.0)
    return distance ** (-exponent)


def _rayleigh(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def generate_channels(
    geometry: Geometry,
    config: ChannelConfig,
    rng: np.random.Generator,
) -> ChannelState:
    """Generate direct and RIS-assisted channels."""
    bs_positions = geometry.bs_positions
    ue_positions = geometry.ue_positions
    ris_positions = geometry.ris_positions

    user_cells = geometry.ue_cells
    ris_cells = geometry.ris_cells

    k_total = ue_positions.shape[0]
    l_total = ris_positions.shape[0]
    nt = config.bs_antennas
    n_elements = config.ris_elements

    h_direct = np.zeros((k_total, nt), dtype=np.complex128)
    for k in range(k_total):
        bs_pos = bs_positions[user_cells[k]]
        distance = np.linalg.norm(ue_positions[k] - bs_pos)
        pl = _pathloss(distance, config.pathloss_bs_ue)
        h_direct[k] = np.sqrt(pl) * _rayleigh((nt,), rng)

    g_bs_ris = np.zeros((l_total, nt, n_elements), dtype=np.complex128)
    for l in range(l_total):
        bs_pos = bs_positions[ris_cells[l]]
        distance = np.linalg.norm(ris_positions[l] - bs_pos)
        pl = _pathloss(distance, config.pathloss_bs_ris_same)
        g_bs_ris[l] = np.sqrt(pl) * _rayleigh((nt, n_elements), rng)

    h_ris_ue = np.zeros((l_total, k_total, n_elements), dtype=np.complex128)
    for l in range(l_total):
        for k in range(k_total):
            distance = np.linalg.norm(ue_positions[k] - ris_positions[l])
            pl = _pathloss(distance, config.pathloss_ris_ue)
            h_ris_ue[l, k] = np.sqrt(pl) * _rayleigh((n_elements,), rng)

    noise_power_linear = 10 ** ((config.noise_power_dbm - 30.0) / 10.0)

    return ChannelState(
        h_direct=h_direct,
        g_bs_ris=g_bs_ris,
        h_ris_ue=h_ris_ue,
        noise_power=noise_power_linear,
    )
