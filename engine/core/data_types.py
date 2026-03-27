"""Shared data types for the engine pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ScanMetadata:
    ac_vents: list[Coordinate] = field(default_factory=list)
    legacy_servers: list[Coordinate] = field(default_factory=list)
    human_workspaces: list[Coordinate] = field(default_factory=list)
