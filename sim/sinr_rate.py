"""Compute SINR and rates for multi-user transmission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RateResult:
    sinr: np.ndarray
    rates: np.ndarray
    sum_rate: float
    beamformers: np.ndarray
    power_alloc: np.ndarray


def compute_rates(
    h_eff: np.ndarray,
    total_power: float,
    noise_power: float,
    bandwidth: float,
) -> RateResult:
    """Compute multi-user SINR and rates with MRT beamforming."""
    k_total, nt = h_eff.shape
    power_alloc = np.full(k_total, total_power / k_total)

    norms = np.linalg.norm(h_eff, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    beamformers = h_eff / norms[:, None]

    gains = np.abs(h_eff @ beamformers.conj().T) ** 2
    signal = power_alloc * np.diag(gains)
    interference = gains @ power_alloc - signal
    sinr = signal / (interference + noise_power)
    rates = bandwidth * np.log2(1.0 + sinr)

    return RateResult(
        sinr=sinr,
        rates=rates,
        sum_rate=float(np.sum(rates)),
        beamformers=beamformers,
        power_alloc=power_alloc,
    )
