from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .materials import Material


@dataclass
class RayBundle:
    position: np.ndarray
    angle: np.ndarray
    weight: np.ndarray
    alive: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.angle = np.asarray(self.angle, dtype=float)
        self.weight = np.asarray(self.weight, dtype=float)
        if self.position.ndim != 2 or self.position.shape[1] not in (1, 2):
            raise ValueError("position must have shape (n, 1) or (n, 2)")
        if self.angle.shape != self.position.shape or self.weight.shape != (
            len(self.position),
        ):
            raise ValueError("inconsistent ray array shapes")
        if np.any(~np.isfinite(self.position)) or np.any(~np.isfinite(self.angle)):
            raise ValueError("ray coordinates must be finite")
        if np.any(~np.isfinite(self.weight)) or np.any(self.weight < 0):
            raise ValueError("weights must be finite and nonnegative")
        if self.alive is None:
            self.alive = np.ones(len(self.position), dtype=bool)
        else:
            self.alive = np.asarray(self.alive, dtype=bool)
            if self.alive.shape != (len(self.position),):
                raise ValueError("alive must have shape (n,)")
        self.weight = np.where(self.alive, self.weight, 0.0)

    @property
    def dimension(self) -> int:
        return self.position.shape[1]

    def copy(self):
        assert self.alive is not None
        return RayBundle(
            self.position.copy(),
            self.angle.copy(),
            self.weight.copy(),
            self.alive.copy(),
        )


@dataclass(frozen=True)
class LensStack:
    z: float
    material: Material
    count: float
    radius: float
    aperture: float
    apex_thickness: float = 0.0
    pitch: float = 0.0
    shape: str = "parabolic"
    axis: Optional[str] = None

    def __post_init__(self):
        if not np.isfinite(self.z):
            raise ValueError("z must be finite")
        if (
            not np.isfinite(self.count)
            or self.count < 0.5
            or not np.isclose(self.count * 2.0, round(self.count * 2.0))
        ):
            raise ValueError("count must be positive in 0.5-lens steps")
        dimensions = np.asarray(
            (self.radius, self.aperture, self.apex_thickness, self.pitch), dtype=float
        )
        if (
            np.any(~np.isfinite(dimensions))
            or self.radius <= 0
            or self.aperture <= 0
            or self.apex_thickness < 0
            or self.pitch < 0
        ):
            raise ValueError("invalid lens dimensions")
        if self.shape not in ("parabolic", "spherical"):
            raise ValueError("unsupported lens shape")
        if self.shape == "spherical" and self.aperture > 1.9 * self.radius:
            raise ValueError("spherical lens aperture must not exceed 95% of diameter")
        if self.axis not in (None, "2D", "x", "y"):
            raise ValueError("focus axis must be '2D', 'x', or 'y'")

    @property
    def centers(self) -> np.ndarray:
        offsets = np.arange(self.element_count, dtype=float) * self.pitch
        return np.asarray(
            self.z + offsets - 0.5 * offsets[-1],
            dtype=float,
        )

    @property
    def element_count(self) -> int:
        return int(np.ceil(self.count))

    def surface_factor(self, lens_index: int) -> float:
        if lens_index == self.element_count - 1 and self.count % 1.0 == 0.5:
            return 0.5
        return 1.0


@dataclass(frozen=True)
class Detector:
    z: float
    aperture: Optional[float] = None

    def __post_init__(self):
        if not np.isfinite(self.z) or (
            self.aperture is not None
            and (not np.isfinite(self.aperture) or self.aperture <= 0)
        ):
            raise ValueError("invalid detector")


@dataclass(frozen=True)
class Aperture:
    z: float
    shape: str
    size_x: float
    size_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self):
        values = np.asarray(
            (self.z, self.size_x, self.size_y, self.offset_x, self.offset_y),
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or self.size_x <= 0 or self.size_y <= 0:
            raise ValueError("invalid aperture dimensions")
        if self.shape not in ("rectangular", "elliptical"):
            raise ValueError("unsupported aperture shape")


@dataclass
class Trajectory:
    z: np.ndarray
    position: np.ndarray
    angle: np.ndarray
    weight: np.ndarray
    labels: tuple[str, ...]


@dataclass
class SimulationResult:
    source: RayBundle
    detector: RayBundle
    detector_z: float
    trajectory: Trajectory
    envelope: dict[str, np.ndarray]
    focus: dict[str, float]
    initial_weight: float
    aperture_loss: float
    absorption_loss: float
    metadata: dict[str, object] = field(default_factory=dict)
    profile: Optional[np.ndarray] = None
    stats: Optional[dict[str, object]] = None
    profile_coordinates: Optional[np.ndarray] = None
    profile_extent: Optional[tuple[float, float, float, float]] = None
