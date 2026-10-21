from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.special import ndtr

from .models import RayBundle

ArrayLike = Union[float, Sequence[float], np.ndarray]


def _vector(
    value: ArrayLike, dimension: int, name: str, nonnegative: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(dimension, float(array))
    if (
        array.shape != (dimension,)
        or np.any(~np.isfinite(array))
        or (nonnegative and np.any(array < 0))
    ):
        raise ValueError(f"invalid {name}")
    return array


@dataclass(frozen=True)
class Source:
    z: float = 0.0
    dimension: int = 1
    spatial: str = "point"
    size: ArrayLike = 0.0
    divergence: str = "uniform"
    angular_size: ArrayLike = 0.0
    center: ArrayLike = 0.0
    mean_angle: ArrayLike = 0.0
    spatial_bounds: Optional[Sequence[Sequence[float]]] = None
    angular_bounds: Optional[Sequence[Sequence[float]]] = None

    def __post_init__(self):
        if self.dimension not in (1, 2) or not np.isfinite(self.z):
            raise ValueError("source dimension must be 1 or 2 and z finite")
        if self.spatial not in ("point", "uniform", "gaussian"):
            raise ValueError("unsupported spatial distribution")
        if self.divergence not in ("uniform", "gaussian"):
            raise ValueError("unsupported angular distribution")
        _vector(self.size, self.dimension, "size", True)
        _vector(self.angular_size, self.dimension, "angular_size", True)
        _vector(self.center, self.dimension, "center")
        _vector(self.mean_angle, self.dimension, "mean_angle")
        self._validate_support(
            self.spatial,
            _vector(self.size, self.dimension, "size", True),
            _vector(self.center, self.dimension, "center"),
            self.spatial_bounds,
        )
        self._validate_support(
            self.divergence,
            _vector(self.angular_size, self.dimension, "angular_size", True),
            _vector(self.mean_angle, self.dimension, "mean_angle"),
            self.angular_bounds,
        )

    def _bounds(self, value, name):
        if value is None:
            return None
        bounds = np.asarray(value, dtype=float)
        if self.dimension == 1 and bounds.shape == (2,):
            bounds = bounds.reshape(1, 2)
        if (
            bounds.shape != (self.dimension, 2)
            or np.any(~np.isfinite(bounds))
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError(f"invalid {name}")
        return bounds

    def _sample(self, rng, count, kind, scale, center, bounds):
        if kind == "point":
            values = np.broadcast_to(center, (count, self.dimension)).copy()
        elif kind == "uniform":
            values = rng.uniform(
                center - scale, center + scale, (count, self.dimension)
            )
        else:
            values = rng.normal(center, scale, (count, self.dimension))
        limits = self._bounds(bounds, "bounds")
        if limits is not None:
            for axis in range(self.dimension):
                outside = (values[:, axis] < limits[axis, 0]) | (
                    values[:, axis] > limits[axis, 1]
                )
                attempts = 0
                while np.any(outside) and attempts < 1000:
                    if kind == "uniform":
                        values[outside, axis] = rng.uniform(
                            center[axis] - scale[axis],
                            center[axis] + scale[axis],
                            np.sum(outside),
                        )
                    elif kind == "gaussian":
                        values[outside, axis] = rng.normal(
                            center[axis], scale[axis], np.sum(outside)
                        )
                    else:
                        raise ValueError("point source lies outside bounds")
                    outside = (values[:, axis] < limits[axis, 0]) | (
                        values[:, axis] > limits[axis, 1]
                    )
                    attempts += 1
                if np.any(outside):
                    raise RuntimeError("bounded source sampling did not converge")
        return values

    def sample(self, count: int, seed: Optional[int] = None) -> RayBundle:
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        rng = np.random.default_rng(seed)
        spatial_scale = _vector(self.size, self.dimension, "size", True)
        angular_scale = _vector(self.angular_size, self.dimension, "angular_size", True)
        spatial_center = _vector(self.center, self.dimension, "center")
        angular_center = _vector(self.mean_angle, self.dimension, "mean_angle")
        self._validate_support(
            self.spatial, spatial_scale, spatial_center, self.spatial_bounds
        )
        self._validate_support(
            self.divergence, angular_scale, angular_center, self.angular_bounds
        )
        position = self._sample(
            rng, count, self.spatial, spatial_scale, spatial_center, self.spatial_bounds
        )
        angle = self._sample(
            rng,
            count,
            self.divergence,
            angular_scale,
            angular_center,
            self.angular_bounds,
        )
        return RayBundle(position, angle, np.ones(count, dtype=float))

    def _validate_support(self, kind, scale, center, bounds):
        limits = self._bounds(bounds, "bounds")
        if limits is None:
            return
        if kind == "uniform" and np.any(
            (center + scale <= limits[:, 0]) | (center - scale >= limits[:, 1])
        ):
            raise ValueError("bounds do not intersect distribution support")
        fixed = scale == 0
        if np.any(fixed & ((center < limits[:, 0]) | (center > limits[:, 1]))):
            raise ValueError("zero-width distribution lies outside bounds")
        if kind == "gaussian":
            variable = ~fixed
            probability = np.ones(self.dimension)
            probability[variable] = ndtr(
                (limits[variable, 1] - center[variable]) / scale[variable]
            ) - ndtr((limits[variable, 0] - center[variable]) / scale[variable])
            if np.any(~np.isfinite(probability)) or np.any(probability <= 1e-12):
                raise ValueError("bounds have negligible distribution probability")
