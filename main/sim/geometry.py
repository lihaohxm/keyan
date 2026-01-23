"""Geometry generation for BS, RIS, and UE positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from sim.config import GeometryConfig


@dataclass
class Geometry:
    bs_positions: np.ndarray
    ris_positions: np.ndarray
    ue_positions: np.ndarray
    ris_cells: np.ndarray
    ue_cells: np.ndarray


def _random_points(center: Tuple[float, float], radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    angles = rng.uniform(0.0, 2 * np.pi, count)
    radii = rng.uniform(0.2 * radius, radius, count)
    xs = center[0] + radii * np.cos(angles)
    ys = center[1] + radii * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def generate_geometry(config: GeometryConfig, rng: np.random.Generator) -> Geometry:
    """Generate positions for BSs, RISs, and users."""
    bs_positions = np.asarray(config.bs_positions, dtype=float)

    ris_positions = []
    ris_cells = []
    ue_positions = []
    ue_cells = []

    for cell_idx, bs_pos in enumerate(bs_positions):
        ris_positions.append(
            _random_points(tuple(bs_pos), config.ris_radius, config.ris_per_cell, rng)
        )
        ris_cells.extend([cell_idx] * config.ris_per_cell)

        ue_positions.append(
            _random_points(tuple(bs_pos), config.cell_radius, config.users_per_cell, rng)
        )
        ue_cells.extend([cell_idx] * config.users_per_cell)

    ris_positions = np.vstack(ris_positions)
    ue_positions = np.vstack(ue_positions)

    return Geometry(
        bs_positions=bs_positions,
        ris_positions=ris_positions,
        ue_positions=ue_positions,
        ris_cells=np.asarray(ris_cells, dtype=int),
        ue_cells=np.asarray(ue_cells, dtype=int),
    )
