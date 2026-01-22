"""Configuration defaults for the multi-RIS semantic communication simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GeometryConfig:
    cell_count: int = 2
    bs_positions: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (0.0, 320.0)]
    )
    users_per_cell: int = 8
    ris_per_cell: int = 6
    cell_radius: float = 150.0
    ris_radius: float = 80.0


@dataclass
class ChannelConfig:
    bs_antennas: int = 4
    ris_elements: int = 36
    noise_power_dbm: float = -70.0
    pathloss_bs_ue: float = 3.2
    pathloss_ris_ue: float = 2.0
    pathloss_bs_ris_same: float = 2.0
    pathloss_bs_ris_cross: float = 2.6


@dataclass
class SemanticConfig:
    method: str = "exp"
    a: float = 0.8
    b: float = 0.15
    c: float = 1.0
    table: dict | None = None


@dataclass
class QoEConfig:
    payload_ratio: float = 1.0
    delay_thresholds_ms: Tuple[float, float] = (10.0, 30.0)
    semantic_threshold: float = 0.2
    beta_delay: float = 1.0
    beta_semantic: float = 1.0
    hard_ratio: float = 0.5
    penalty_delay: float = 10.0
    penalty_semantic: float = 10.0
    weights: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "delay_sensitive": (0.8, 0.2),
            "balanced": (0.5, 0.5),
            "semantic_sensitive": (0.2, 0.8),
        }
    )


@dataclass
class SimulationConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    qoe: QoEConfig = field(default_factory=QoEConfig)
    bandwidth_hz: float = 1e6
    total_power_dbw: float = -10.0
    ris_capacity: int = 4
    payload_symbols: int = 8
    seed: int = 1


DEFAULT_POWER_DBW: List[float] = [-10.0, -5.0, 0.0]
