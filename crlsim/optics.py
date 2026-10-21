import numpy as np

from .models import Aperture, LensStack, RayBundle


def drift(bundle: RayBundle, distance: float) -> None:
    bundle.position += np.tan(bundle.angle) * distance


def _coordinates(
    bundle: RayBundle, stack: LensStack
) -> tuple[np.ndarray, tuple[int, ...]]:
    if stack.axis in ("x", "y") and bundle.dimension == 2:
        index = 0 if stack.axis == "x" else 1
        radial2 = bundle.position[:, index] ** 2
        active: tuple[int, ...] = (index,)
    else:
        radial2 = np.sum(bundle.position**2, axis=1)
        active = tuple(range(bundle.dimension))
    return radial2, active


def apply_lens(
    bundle: RayBundle,
    stack: LensStack,
    delta: float,
    mu: float,
    surface_factor: float = 1.0,
) -> tuple[float, float]:
    assert bundle.alive is not None
    radial2, active = _coordinates(bundle, stack)
    before = float(np.sum(bundle.weight))
    if stack.axis in ("x", "y") and bundle.dimension == 2:
        powered_axis = 0 if stack.axis == "x" else 1
        inside = np.abs(bundle.position[:, powered_axis]) <= stack.aperture * 0.5
    else:
        aperture_radius2 = np.sum(bundle.position**2, axis=1)
        inside = aperture_radius2 <= (stack.aperture * 0.5) ** 2
    clipped = bundle.alive & ~inside
    bundle.alive[clipped] = False
    bundle.weight[clipped] = 0.0
    aperture_loss = before - float(np.sum(bundle.weight))
    valid = bundle.alive
    if stack.shape == "parabolic":
        thickness = stack.apex_thickness + surface_factor * radial2 / stack.radius
        denominator = np.full(len(bundle.position), stack.radius / surface_factor)
    else:
        root = np.sqrt(np.maximum(stack.radius**2 - radial2, 0.0))
        thickness = stack.apex_thickness + surface_factor * 2.0 * (stack.radius - root)
        denominator = root / surface_factor
    pre_absorption = float(np.sum(bundle.weight))
    bundle.weight[valid] *= np.exp(-mu * thickness[valid])
    absorption_loss = pre_absorption - float(np.sum(bundle.weight))
    for axis in active:
        bundle.angle[valid, axis] -= (
            2.0 * delta * bundle.position[valid, axis] / denominator[valid]
        )
    return aperture_loss, absorption_loss


def apply_aperture(bundle: RayBundle, aperture: Aperture) -> float:
    assert bundle.alive is not None
    x = bundle.position[:, 0] - aperture.offset_x
    if bundle.dimension == 2:
        y = bundle.position[:, 1] - aperture.offset_y
    else:
        y = np.zeros(len(bundle.position))
    if aperture.shape == "rectangular":
        inside = (np.abs(x) <= aperture.size_x * 0.5) & (
            np.abs(y) <= aperture.size_y * 0.5
        )
    else:
        inside = (x / (aperture.size_x * 0.5)) ** 2 + (
            y / (aperture.size_y * 0.5)
        ) ** 2 <= 1.0
    rejected = bundle.alive & ~inside
    loss = float(np.sum(bundle.weight[rejected]))
    bundle.alive[rejected] = False
    bundle.weight[rejected] = 0.0
    return loss
