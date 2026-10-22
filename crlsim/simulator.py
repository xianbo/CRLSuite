from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import (
    Aperture,
    Detector,
    LensStack,
    RayBundle,
    SimulationResult,
    Trajectory,
)
from .optics import apply_aperture, apply_lens, drift
from .sources import Source


@dataclass
class Simulator:
    energy_keV: float

    def __post_init__(self):
        if not np.isfinite(self.energy_keV) or self.energy_keV <= 0:
            raise ValueError("energy must be positive keV")

    def run(
        self,
        source: Source | RayBundle,
        stacks: Sequence[LensStack],
        detector: Detector,
        rays: int = 100000,
        seed: int | None = None,
        source_plane: float = 0.0,
        apertures: Sequence[Aperture] = (),
    ) -> SimulationResult:
        bundle = (
            source.sample(rays, seed) if isinstance(source, Source) else source.copy()
        )
        source_z = source.z if isinstance(source, Source) else source_plane
        initial = bundle.copy()
        initial_weight = float(np.sum(bundle.weight))
        events = []
        constants = {}
        for stack_index, stack in enumerate(stacks):
            constants[stack_index] = stack.material.optical_constants(self.energy_keV)
            for lens_index, z in enumerate(stack.centers):
                events.append(
                    (
                        float(z),
                        1,
                        stack_index,
                        lens_index,
                        stack.surface_factor(lens_index),
                    )
                )
        events.append((float(detector.z), 3, -1, -1, 0.0))
        for aperture_index, aperture in enumerate(apertures):
            events.append((float(aperture.z), 2, aperture_index, -1, 0.0))
        events.sort(key=lambda event: (event[0], event[1]))
        if events and events[0][0] < source_z:
            raise ValueError("optical events cannot precede the source plane")
        z_values = [float(source_z)]
        positions = [bundle.position.copy()]
        angles = [bundle.angle.copy()]
        weights = [bundle.weight.copy()]
        labels = ["source"]
        current_z = float(source_z)
        aperture_loss = 0.0
        absorption_loss = 0.0
        detector_bundle = None
        detector_aperture_loss = 0.0
        detector_absorption_loss = 0.0
        for z, kind, stack_index, lens_index, surface_factor in events:
            drift(bundle, z - current_z)
            current_z = z
            if kind == 3:
                detector_bundle = bundle.copy()
                if detector.aperture is not None:
                    radial2 = np.sum(detector_bundle.position**2, axis=1)
                    rejected = detector_bundle.alive & (
                        radial2 > (detector.aperture * 0.5) ** 2
                    )
                    loss = float(np.sum(detector_bundle.weight[rejected]))
                    detector_bundle.alive[rejected] = False
                    detector_bundle.weight[rejected] = 0.0
                    detector_aperture_loss = aperture_loss + loss
                else:
                    detector_aperture_loss = aperture_loss
                detector_absorption_loss = absorption_loss
                snapshot = detector_bundle
                label = "detector"
            elif kind == 1:
                stack = stacks[stack_index]
                optical = constants[stack_index]
                lost_aperture, lost_absorption = apply_lens(
                    bundle,
                    stack,
                    optical.delta,
                    optical.mu,
                    surface_factor,
                )
                aperture_loss += lost_aperture
                absorption_loss += lost_absorption
                snapshot = bundle
                label = f"stack_{stack_index}_lens_{lens_index}"
            else:
                z_values.append(z)
                positions.append(bundle.position.copy())
                angles.append(bundle.angle.copy())
                weights.append(bundle.weight.copy())
                labels.append(f"aperture_{stack_index}_before")
                loss = apply_aperture(bundle, apertures[stack_index])
                aperture_loss += loss
                snapshot = bundle
                label = f"aperture_{stack_index}"
            z_values.append(z)
            positions.append(snapshot.position.copy())
            angles.append(snapshot.angle.copy())
            weights.append(snapshot.weight.copy())
            labels.append(label)
            if kind == 3:
                break
        trajectory = Trajectory(
            np.asarray(z_values),
            np.stack(positions),
            np.stack(angles),
            np.stack(weights),
            tuple(labels),
        )
        envelope = self._envelope(trajectory)
        if detector_bundle is None:
            raise RuntimeError("detector event was not evaluated")
        focus = self._focus(detector_bundle, detector.z)
        focal_positions = self._focal_positions(trajectory)
        return SimulationResult(
            initial,
            detector_bundle,
            detector.z,
            trajectory,
            envelope,
            focus,
            initial_weight,
            detector_aperture_loss,
            detector_absorption_loss,
            {
                "energy_keV": self.energy_keV,
                "lens_stacks": len(stacks),
                "attenuation_model": "xraylib CS_Total_CP",
                "mu_total_m_inv": [constants[index].mu for index in range(len(stacks))],
                "focal_positions": focal_positions,
            },
        )

    @staticmethod
    def _envelope(trajectory):
        centroid = []
        sigma = []
        for position, weight in zip(
            trajectory.position, trajectory.weight, strict=False
        ):
            total = np.sum(weight)
            if total <= 0:
                centroid.append(np.full(position.shape[1], np.nan))
                sigma.append(np.full(position.shape[1], np.nan))
            else:
                mean = np.sum(position * weight[:, None], axis=0) / total
                centroid.append(mean)
                sigma.append(
                    np.sqrt(
                        np.sum((position - mean) ** 2 * weight[:, None], axis=0) / total
                    )
                )
        return {
            "z": trajectory.z.copy(),
            "centroid": np.asarray(centroid),
            "sigma": np.asarray(sigma),
            "lower": np.asarray(centroid) - np.asarray(sigma),
            "upper": np.asarray(centroid) + np.asarray(sigma),
        }

    @staticmethod
    def _focus(bundle, z):
        result = {}
        names = ("x", "y")
        valid = bundle.weight > 0
        for axis in range(bundle.dimension):
            p = bundle.position[valid, axis]
            a = bundle.angle[valid, axis]
            w = bundle.weight[valid]
            if np.sum(w) <= 0:
                result[names[axis]] = np.nan
                continue
            mean_p = np.average(p, weights=w)
            mean_a = np.average(a, weights=w)
            variance_a = np.average((a - mean_a) ** 2, weights=w)
            covariance = np.average((p - mean_p) * (a - mean_a), weights=w)
            result[names[axis]] = (
                float(z - covariance / variance_a) if variance_a > 0 else np.nan
            )
        return result

    @staticmethod
    def _focal_positions(trajectory):
        positions = []
        last_by_stack = {}
        for index, label in enumerate(trajectory.labels):
            if not label.startswith("stack_"):
                continue
            stack_label = label.split("_lens_", 1)[0]
            last_by_stack[stack_label] = index
        for stack_label, index in last_by_stack.items():
            bundle = RayBundle(
                trajectory.position[index],
                trajectory.angle[index],
                trajectory.weight[index],
            )
            positions.append(
                {
                    "label": stack_label,
                    "z_m": float(trajectory.z[index]),
                    "focus_m": Simulator._focus(bundle, trajectory.z[index]),
                }
            )
        return positions


def simulate(
    source,
    stacks,
    detector,
    energy_keV,
    rays=100000,
    seed=None,
    source_plane=0.0,
    apertures=(),
):
    return Simulator(energy_keV).run(
        source, stacks, detector, rays, seed, source_plane, apertures
    )
