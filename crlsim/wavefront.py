from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import ndtr

from .models import Aperture, Detector, LensStack
from .profiles import (
    gaussian_fit_fwhm,
    interpolated_fwhm,
    profile_crop_slice,
    statistics2d,
)
from .sources import Source


@dataclass
class WavefrontResult:
    profile: np.ndarray
    coordinates_x: np.ndarray
    coordinates_y: Optional[np.ndarray]
    detector_z: float
    stats: dict[str, object]
    transmission: float
    metadata: dict[str, object] = field(default_factory=dict)
    profile_coordinates: Optional[np.ndarray] = None
    profile_extent: Optional[tuple[float, float, float, float]] = None


def _axis(size: float, points: int) -> np.ndarray:
    return np.asarray((np.arange(points) - points // 2) * (size / points))


def _point_source_field(
    source: Source,
    points: int,
    spacing: float,
    wavelength: float,
    source_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    angular_size = np.broadcast_to(
        np.asarray(source.angular_size, dtype=float), source.dimension
    )
    mean_angle = np.broadcast_to(
        np.asarray(source.mean_angle, dtype=float), source.dimension
    )
    bounds = source._bounds(source.angular_bounds, "angular_bounds")
    frequency = np.fft.fftfreq(points, spacing)

    def amplitude(index: int) -> np.ndarray:
        theta = wavelength * frequency
        centered = theta - mean_angle[index]
        if source.divergence == "uniform":
            spectrum = (np.abs(centered) <= angular_size[index]).astype(float)
        elif angular_size[index] > 0:
            spectrum = np.exp(-(centered**2) / (4.0 * angular_size[index] ** 2))
        else:
            spectrum = np.zeros(points)
            spectrum[int(np.argmin(np.abs(centered)))] = 1.0
        if bounds is not None:
            spectrum *= (theta >= bounds[index, 0]) & (theta <= bounds[index, 1])
        return np.asarray(spectrum)

    spectrum_x = amplitude(0)
    if source_offset is not None:
        spectrum_x = spectrum_x * np.exp(-2j * np.pi * frequency * source_offset[0])
    if source.dimension == 1:
        return np.asarray(np.fft.fftshift(np.fft.ifft(spectrum_x)))
    spectrum_y = amplitude(1)
    if source_offset is not None:
        spectrum_y = spectrum_y * np.exp(-2j * np.pi * frequency * source_offset[1])
    return np.asarray(
        np.fft.fftshift(np.fft.ifft2(spectrum_y[:, None] * spectrum_x[None, :]))
    )


def _magnification(
    source: Source,
    stacks: list[LensStack],
    detector: Detector,
    energy_keV: float,
) -> np.ndarray:
    matrices = np.repeat(np.eye(2)[None, :, :], source.dimension, axis=0)
    lens_events = []
    for stack in stacks:
        delta = stack.material.optical_constants(energy_keV).delta
        for lens_index, z in enumerate(stack.centers):
            lens_events.append(
                (float(z), stack, stack.surface_factor(lens_index), delta)
            )
    lens_events.sort(key=lambda event: event[0])
    current_z = source.z
    for z, stack, factor, delta in lens_events:
        if z >= detector.z:
            break
        drift = np.array(((1.0, z - current_z), (0.0, 1.0)))
        matrices = np.asarray([drift @ matrix for matrix in matrices])
        lens = np.array(((1.0, 0.0), (-2.0 * delta * factor / stack.radius, 1.0)))
        active: tuple[int, ...] = tuple(range(source.dimension))
        if source.dimension > 1 and stack.axis in ("x", "y"):
            active = (0,) if stack.axis == "x" else (1,)
        for index in active:
            matrices[index] = lens @ matrices[index]
        current_z = z
    drift = np.array(((1.0, detector.z - current_z), (0.0, 1.0)))
    matrices = np.asarray([drift @ matrix for matrix in matrices])
    return np.asarray(matrices[:, 0, 0])


def _stack_focal_positions(
    source: Source, stacks: list[LensStack], energy_keV: float
) -> list[dict[str, object]]:
    matrices = np.repeat(np.eye(2)[None, :, :], source.dimension, axis=0)
    events = []
    last_element = {}
    for stack_index, stack in enumerate(stacks):
        delta = stack.material.optical_constants(energy_keV).delta
        for lens_index, z in enumerate(stack.centers):
            events.append(
                (
                    float(z),
                    stack_index,
                    stack,
                    stack.surface_factor(lens_index),
                    delta,
                )
            )
        last_element[stack_index] = stack.element_count - 1
    events.sort(key=lambda event: event[0])
    current_z = source.z
    focal_positions = []
    powered_axes = np.zeros(source.dimension, dtype=bool)
    seen_counts = {index: 0 for index in range(len(stacks))}
    for z, stack_index, stack, factor, delta in events:
        drift = np.array(((1.0, z - current_z), (0.0, 1.0)))
        matrices = np.asarray([drift @ matrix for matrix in matrices])
        lens = np.array(((1.0, 0.0), (-2.0 * delta * factor / stack.radius, 1.0)))
        active: tuple[int, ...] = tuple(range(source.dimension))
        if source.dimension > 1 and stack.axis in ("x", "y"):
            active = (0,) if stack.axis == "x" else (1,)
        for axis in active:
            matrices[axis] = lens @ matrices[axis]
            powered_axes[axis] = True
        if seen_counts[stack_index] == last_element[stack_index]:
            focus = {}
            for axis, name in enumerate(("x", "y")[: source.dimension]):
                d = matrices[axis, 1, 1]
                focus[name] = (
                    float(z - matrices[axis, 0, 1] / d)
                    if powered_axes[axis] and d != 0
                    else np.nan
                )
            focal_positions.append(
                {"label": f"stack_{stack_index}", "z_m": z, "focus_m": focus}
            )
        seen_counts[stack_index] += 1
        current_z = z
    return focal_positions


def _source_kernel(
    source: Source, axis: np.ndarray, magnification: np.ndarray
) -> np.ndarray:
    size = np.broadcast_to(np.asarray(source.size, dtype=float), source.dimension)
    center = np.broadcast_to(np.asarray(source.center, dtype=float), source.dimension)
    bounds = source._bounds(source.spatial_bounds, "spatial_bounds")

    def kernel_axis(index: int) -> np.ndarray:
        scale = float(magnification[index])
        image_center = scale * center[index]
        if source.spatial == "point" or scale == 0 or size[index] == 0:
            kernel = np.zeros_like(axis)
            kernel[int(np.argmin(np.abs(axis - image_center)))] = 1.0
            return np.asarray(kernel)
        source_coordinate = axis / scale
        if source.spatial == "uniform":
            kernel = (np.abs(source_coordinate - center[index]) <= size[index]).astype(
                float
            )
        else:
            kernel = np.exp(
                -((source_coordinate - center[index]) ** 2) / (2.0 * size[index] ** 2)
            )
        if bounds is not None:
            kernel *= (source_coordinate >= bounds[index, 0]) & (
                source_coordinate <= bounds[index, 1]
            )
        total = np.sum(kernel)
        return np.asarray(kernel / total if total > 0 else kernel)

    kernel_x = kernel_axis(0)
    if source.dimension == 1:
        return kernel_x
    return np.asarray(kernel_axis(1)[:, None] * kernel_x[None, :])


def _field_phase_space(
    field: np.ndarray, axis: np.ndarray, wavelength: float, detector_z: float
) -> dict[str, object]:
    intensity = np.abs(field) ** 2
    wave_number = 2.0 * np.pi / wavelength
    spacing = axis[1] - axis[0]
    results: dict[str, object] = {}
    dimensions = 1 if field.ndim == 1 else 2
    for dimension, name in enumerate(("x", "y")[:dimensions]):
        derivative_axis = 0 if field.ndim == 1 or dimension == 1 else 1
        local_angle = _phase_gradient(field, spacing, derivative_axis) / wave_number
        coordinate = (
            axis
            if field.ndim == 1
            else (axis[None, :] if dimension == 0 else axis[:, None])
        )
        total = np.sum(intensity)
        mean_x = np.sum(intensity * coordinate) / total
        mean_angle = np.sum(intensity * local_angle) / total
        angle_variance = np.sum(intensity * (local_angle - mean_angle) ** 2) / total
        covariance = (
            np.sum(intensity * (coordinate - mean_x) * (local_angle - mean_angle))
            / total
        )
        virtual_source = (
            detector_z - covariance / angle_variance if angle_variance > 0 else np.nan
        )
        spectrum = np.abs(np.fft.fftn(field)) ** 2
        frequency = np.fft.fftfreq(field.shape[derivative_axis], spacing)
        spectral_marginal = np.sum(
            spectrum,
            axis=tuple(
                axis_index
                for axis_index in range(field.ndim)
                if axis_index != derivative_axis
            ),
        )
        spectral_total = np.sum(spectral_marginal)
        theta = wavelength * frequency
        spectral_mean = np.sum(spectral_marginal * theta) / spectral_total
        spectral_variance = (
            np.sum(spectral_marginal * (theta - spectral_mean) ** 2) / spectral_total
        )
        results[f"wavefront_divergence_{name}"] = float(np.sqrt(spectral_variance))
        order = np.argsort(theta)
        results[f"wavefront_divergence_fwhm_{name}"] = interpolated_fwhm(
            theta[order], spectral_marginal[order]
        )
        results[f"wavefront_divergence_gaussian_fwhm_{name}"] = gaussian_fit_fwhm(
            theta[order], spectral_marginal[order]
        )
        results[f"wavefront_virtual_source_{name}"] = float(virtual_source)
    return results


def _field_waist_positions(
    field: np.ndarray, axis: np.ndarray, wavelength: float, z: float
) -> dict[str, float]:
    intensity = np.abs(field) ** 2
    wave_number = 2.0 * np.pi / wavelength
    spacing = axis[1] - axis[0]
    focus = {}
    dimensions = 1 if field.ndim == 1 else 2
    for dimension, name in enumerate(("x", "y")[:dimensions]):
        derivative_axis = 0 if field.ndim == 1 or dimension == 1 else 1
        local_angle = _phase_gradient(field, spacing, derivative_axis) / wave_number
        coordinate = (
            axis
            if field.ndim == 1
            else (axis[None, :] if dimension == 0 else axis[:, None])
        )
        total = np.sum(intensity)
        mean_x = np.sum(intensity * coordinate) / total
        mean_angle = np.sum(intensity * local_angle) / total
        variance_x = np.sum(intensity * (coordinate - mean_x) ** 2) / total
        angle_variance = np.sum(intensity * (local_angle - mean_angle) ** 2) / total
        covariance = (
            np.sum(intensity * (coordinate - mean_x) * (local_angle - mean_angle))
            / total
        )
        threshold = (
            64.0
            * np.finfo(float).eps
            * np.sqrt(max(float(variance_x * angle_variance), 0.0))
        )
        focus[name] = (
            float(z - covariance / angle_variance)
            if (angle_variance > 0.0 and abs(covariance) > threshold)
            else np.nan
        )
    return focus


def _phase_gradient(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    forward = np.roll(field, -1, axis=axis)
    backward = np.roll(field, 1, axis=axis)
    gradient = np.angle(forward * np.conj(backward)) / (2.0 * spacing)
    first: list[Union[slice, int]] = [slice(None)] * field.ndim
    second: list[Union[slice, int]] = [slice(None)] * field.ndim
    last: list[Union[slice, int]] = [slice(None)] * field.ndim
    previous: list[Union[slice, int]] = [slice(None)] * field.ndim
    first[axis], second[axis] = 0, 1
    last[axis], previous[axis] = -1, -2
    gradient[tuple(first)] = (
        np.angle(field[tuple(second)] * np.conj(field[tuple(first)])) / spacing
    )
    gradient[tuple(last)] = (
        np.angle(field[tuple(last)] * np.conj(field[tuple(previous)])) / spacing
    )
    return np.asarray(gradient)


def _propagate(
    field: np.ndarray, spacing: float, wavelength: float, distance: float
) -> np.ndarray:
    if distance == 0:
        return field
    if field.ndim == 1:
        frequency2 = np.fft.fftfreq(field.shape[0], spacing) ** 2
    else:
        fx = np.fft.fftfreq(field.shape[1], spacing)
        fy = np.fft.fftfreq(field.shape[0], spacing)
        frequency2 = fy[:, None] ** 2 + fx[None, :] ** 2
    transfer = np.exp(-1j * np.pi * wavelength * distance * frequency2)
    return np.asarray(np.fft.ifftn(np.fft.fftn(field) * transfer))


def _lens_transfer(
    axes: tuple[np.ndarray, ...],
    stack: LensStack,
    delta: float,
    mu: float,
    factor: float,
    wavelength: float,
) -> np.ndarray:
    grids = np.meshgrid(*axes, indexing="xy") if len(axes) == 2 else axes
    x = grids[0]
    y = grids[1] if len(axes) == 2 else np.zeros_like(x)
    aperture_r2 = x**2 + y**2
    if stack.axis in ("x", "y") and len(axes) == 2:
        radial2 = x**2 if stack.axis == "x" else y**2
    else:
        radial2 = aperture_r2
    if stack.axis == "x" and len(axes) == 2:
        inside = np.abs(x) <= stack.aperture * 0.5
    elif stack.axis == "y" and len(axes) == 2:
        inside = np.abs(y) <= stack.aperture * 0.5
    else:
        inside = aperture_r2 <= (stack.aperture * 0.5) ** 2
    if stack.shape == "parabolic":
        thickness = stack.apex_thickness + factor * radial2 / stack.radius
    else:
        root = np.sqrt(np.maximum(stack.radius**2 - radial2, 0.0))
        inside &= radial2 < stack.radius**2
        thickness = stack.apex_thickness + factor * 2.0 * (stack.radius - root)
    wave_number = 2.0 * np.pi / wavelength
    return np.asarray(
        inside * np.exp(-0.5 * mu * thickness - 1j * wave_number * delta * thickness)
    )


def _aperture_transfer(axes: tuple[np.ndarray, ...], aperture: Aperture) -> np.ndarray:
    grids = np.meshgrid(*axes, indexing="xy") if len(axes) == 2 else axes
    x = grids[0] - aperture.offset_x
    y = grids[1] - aperture.offset_y if len(axes) == 2 else np.zeros_like(x)
    if aperture.shape == "rectangular":
        return (np.abs(x) <= aperture.size_x * 0.5) & (
            np.abs(y) <= aperture.size_y * 0.5
        )
    return (x / (aperture.size_x * 0.5)) ** 2 + (
        y / (aperture.size_y * 0.5)
    ) ** 2 <= 1.0


def _detector_transfer(axes: tuple[np.ndarray, ...], detector: Detector) -> np.ndarray:
    if detector.aperture is None:
        return np.ones(tuple(len(axis) for axis in axes[::-1]), dtype=bool)
    if len(axes) == 1:
        return np.abs(axes[0]) <= detector.aperture * 0.5
    x, y = np.meshgrid(*axes, indexing="xy")
    return np.array(x**2 + y**2 <= (detector.aperture * 0.5) ** 2, dtype=bool)


def _centered_convolve(intensity: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full = fftconvolve(intensity, kernel, mode="full")
    starts = tuple(size // 2 for size in kernel.shape)
    slices = tuple(
        slice(start, start + size) for start, size in zip(starts, intensity.shape)
    )
    result: np.ndarray = np.asarray(full[slices])
    return result


def _angular_capture(
    source: Source, wavelength: float, spacing: float
) -> tuple[float, bool]:
    sizes = np.broadcast_to(
        np.asarray(source.angular_size, dtype=float), source.dimension
    )
    means = np.broadcast_to(
        np.asarray(source.mean_angle, dtype=float), source.dimension
    )
    bounds = source._bounds(source.angular_bounds, "angular_bounds")
    nyquist = wavelength / (2.0 * spacing)
    captured = 1.0
    for index in range(source.dimension):
        lower = -np.inf if bounds is None else bounds[index, 0]
        upper = np.inf if bounds is None else bounds[index, 1]
        clipped_lower = max(lower, -nyquist)
        clipped_upper = min(upper, nyquist)
        if sizes[index] == 0:
            fraction = float(clipped_lower <= means[index] <= clipped_upper)
        elif source.divergence == "uniform":
            support_lower = max(lower, means[index] - sizes[index])
            support_upper = min(upper, means[index] + sizes[index])
            denominator = max(support_upper - support_lower, 0.0)
            numerator = max(
                min(support_upper, nyquist) - max(support_lower, -nyquist), 0.0
            )
            fraction = numerator / denominator if denominator > 0 else 0.0
        else:
            denominator = ndtr((upper - means[index]) / sizes[index]) - ndtr(
                (lower - means[index]) / sizes[index]
            )
            numerator = ndtr((clipped_upper - means[index]) / sizes[index]) - ndtr(
                (clipped_lower - means[index]) / sizes[index]
            )
            fraction = float(numerator / denominator) if denominator > 0 else 0.0
        captured *= fraction
    return float(captured), captured < 1.0 - 1e-6


def simulate_wavefront(
    source: Source,
    stacks: list[LensStack],
    apertures: list[Aperture],
    detector: Detector,
    energy_keV: float,
    points: int = 512,
    window_m: float = 2e-3,
    source_offset: Optional[np.ndarray] = None,
    apply_source_convolution: bool = True,
) -> WavefrontResult:
    if points < 32 or points & (points - 1):
        raise ValueError("wavefront points must be a power of two and at least 32")
    if source.dimension == 2 and points > 2048:
        raise ValueError(
            "2D wavefront points cannot exceed 2048; reduce the window to improve "
            "sampling"
        )
    if window_m <= 0:
        raise ValueError("wavefront window must be positive")
    wavelength = 1.239841984e-9 / energy_keV
    axis = _axis(window_m, points)
    axes = (axis,) if source.dimension == 1 else (axis, axis)
    spacing = window_m / points
    captured_power, angular_aliasing = _angular_capture(source, wavelength, spacing)
    if captured_power <= 0:
        raise ValueError("source divergence is outside the wavefront Nyquist range")
    source_center = np.broadcast_to(
        np.asarray(source.center, dtype=float), source.dimension
    )
    effective_offset = source_offset
    if effective_offset is None and source.spatial == "point":
        effective_offset = source_center
    field = _point_source_field(source, points, spacing, wavelength, effective_offset)
    initial_power = float(np.sum(np.abs(field) ** 2))
    if initial_power <= 0:
        raise ValueError("source divergence is not represented by the wavefront grid")
    field *= np.sqrt(captured_power / initial_power)
    initial_power = captured_power
    aperture_loss = 0.0
    absorption_loss = 0.0
    explicit_aperture_clipped = False
    envelope_z = [float(source.z)]
    envelope_lower: dict[str, list[float]] = {
        name: [] for name in ("x", "y")[: source.dimension]
    }
    envelope_upper: dict[str, list[float]] = {
        name: [] for name in ("x", "y")[: source.dimension]
    }

    def record_envelope() -> None:
        intensity_now = np.abs(field) ** 2
        for axis_index, name in enumerate(("x", "y")[: source.dimension]):
            marginal = (
                intensity_now
                if intensity_now.ndim == 1
                else np.sum(intensity_now, axis=0 if axis_index == 0 else 1)
            )
            if np.any(marginal > 0):
                support = profile_crop_slice(marginal, threshold=0.005, padding=0.0)
                envelope_lower[name].append(float(axis[support.start]))
                envelope_upper[name].append(float(axis[support.stop - 1]))
            else:
                envelope_lower[name].append(np.nan)
                envelope_upper[name].append(np.nan)

    record_envelope()
    events = []
    constants = {}
    stack_last_lens = {}
    for stack_index, stack in enumerate(stacks):
        constants[stack_index] = stack.material.optical_constants(energy_keV)
        for lens_index, z in enumerate(stack.centers):
            events.append((float(z), 1, stack_index, lens_index))
        stack_last_lens[stack_index] = stack.element_count - 1
    for aperture_index, aperture in enumerate(apertures):
        events.append((float(aperture.z), 2, aperture_index, -1))
    events.append((float(detector.z), 0, -1, -1))
    events.sort(key=lambda event: (event[0], event[1]))
    current_z = source.z
    numerical_focal_positions = []
    for z, kind, element_index, sub_index in events:
        if z < current_z:
            raise ValueError("wavefront elements cannot precede the source")
        field = _propagate(field, spacing, wavelength, z - current_z)
        current_z = z
        if kind == 0:
            detector_mask = _detector_transfer(axes, detector)
            power_before = float(np.sum(np.abs(field) ** 2))
            field *= detector_mask
            aperture_loss += power_before - float(np.sum(np.abs(field) ** 2))
            envelope_z.append(z)
            record_envelope()
            break
        if kind == 1:
            stack = stacks[element_index]
            optical = constants[element_index]
            transfer = _lens_transfer(
                axes,
                stack,
                optical.delta,
                optical.mu,
                stack.surface_factor(sub_index),
                wavelength,
            )
            power_before = float(np.sum(np.abs(field) ** 2))
            lens_mask = np.abs(transfer) > 0.0
            masked_field = field * lens_mask
            power_masked = float(np.sum(np.abs(masked_field) ** 2))
            field *= transfer
            power_after = float(np.sum(np.abs(field) ** 2))
            aperture_loss += power_before - power_masked
            absorption_loss += power_masked - power_after
            if sub_index == stack_last_lens[element_index]:
                numerical_focal_positions.append(
                    {
                        "label": f"stack_{element_index}",
                        "z_m": z,
                        "focus_m": _field_waist_positions(field, axis, wavelength, z),
                    }
                )
        else:
            envelope_z.append(z)
            record_envelope()
            aperture_mask = _aperture_transfer(axes, apertures[element_index])
            power_before = float(np.sum(np.abs(field) ** 2))
            field *= aperture_mask
            blocked_power = power_before - float(np.sum(np.abs(field) ** 2))
            aperture_loss += blocked_power
            explicit_aperture_clipped |= blocked_power > 1e-8 * initial_power
        envelope_z.append(z)
        record_envelope()
    point_intensity = np.abs(field) ** 2
    optical_transmission = float(np.sum(point_intensity))
    magnification = _magnification(source, stacks, detector, energy_keV)
    focal_positions = (
        numerical_focal_positions
        if explicit_aperture_clipped
        else _stack_focal_positions(source, stacks, energy_keV)
    )
    if apply_source_convolution and source.spatial != "point":
        source_kernel = _source_kernel(source, axis, magnification)
        intensity = _centered_convolve(point_intensity, source_kernel)
    else:
        intensity = point_intensity.copy()
    intensity = np.maximum(intensity, 0.0)
    intensity *= _detector_transfer(axes, detector)
    convolved_power = float(np.sum(intensity))
    convolution_boundary_loss = max(optical_transmission - convolved_power, 0.0)
    transmission = convolved_power
    if source.dimension == 1:
        total = float(np.sum(intensity))
        centroid = float(np.sum(axis * intensity) / total)
        sigma = float(np.sqrt(np.sum(intensity * (axis - centroid) ** 2) / total))
        stats: dict[str, object] = {
            "centroid": centroid,
            "sigma": sigma,
            "rms": float(np.sqrt(np.sum(intensity * axis**2) / total)),
            "fwhm": interpolated_fwhm(axis, intensity),
            "gaussian_fwhm": gaussian_fit_fwhm(axis, intensity),
            "weight": total,
            "transmission": transmission,
            "aperture_loss": aperture_loss,
            "absorption_loss": absorption_loss,
            "angular_spectrum_loss": 1.0 - captured_power,
            "convolution_boundary_loss": convolution_boundary_loss,
        }
        stats.update(_field_phase_space(field, axis, wavelength, detector.z))
        pixel_size = window_m / points
        fwhm_value = stats["fwhm"]
        if not isinstance(fwhm_value, (int, float, np.floating)):
            raise RuntimeError("Invalid wavefront FWHM statistic")
        fwhm_pixels = float(fwhm_value) / pixel_size
        stats["wave_pixel_size"] = pixel_size
        stats["fwhm_pixels"] = fwhm_pixels
        if not np.isfinite(fwhm_pixels) or fwhm_pixels < 4.0:
            stats["sampling_warning"] = (
                "Under-resolved wavefront result: increase Wave points or reduce "
                "Wave window until FWHM pixels is at least 4"
            )
        if angular_aliasing:
            stats["angular_sampling_warning"] = (
                "Wavefront grid does not capture the full source angular spectrum; "
                "reduce the grid spacing"
            )
        return WavefrontResult(
            intensity,
            axis,
            None,
            detector.z,
            stats,
            transmission,
            {
                "engine": "wavefront",
                "wavelength_m": wavelength,
                "window_m": window_m,
                "points": points,
                "source_model": (
                    "point-source angular spectrum plus intensity convolution"
                ),
                "magnification": magnification,
                "point_source_transmission": optical_transmission,
                "captured_angular_power": captured_power,
                "aperture_loss": aperture_loss,
                "absorption_loss": absorption_loss,
                "convolution_boundary_loss": convolution_boundary_loss,
                "focal_position_method": (
                    "numerical covariance after explicit aperture clipping"
                    if explicit_aperture_clipped
                    else "ABCD (lens-aperture diffraction not included)"
                ),
                "focal_positions": focal_positions,
                "boundary_z": envelope_z,
                "boundary_lower_x": envelope_lower["x"],
                "boundary_upper_x": envelope_upper["x"],
            },
            profile_coordinates=axis,
        )
    x, y = np.meshgrid(axis, axis, indexing="xy")
    stats = statistics2d(
        np.column_stack((x.ravel(), y.ravel())),
        intensity.ravel(),
        initial_weight=initial_power,
    )
    stats.update(_field_phase_space(field, axis, wavelength, detector.z))
    pixel_size = window_m / points
    x_profile = np.sum(intensity, axis=0)
    y_profile = np.sum(intensity, axis=1)
    x_fwhm = interpolated_fwhm(axis, x_profile)
    y_fwhm = interpolated_fwhm(axis, y_profile)
    stats.update(
        {
            "x_fwhm": x_fwhm,
            "y_fwhm": y_fwhm,
            "x_gaussian_fwhm": gaussian_fit_fwhm(axis, x_profile),
            "y_gaussian_fwhm": gaussian_fit_fwhm(axis, y_profile),
            "wave_pixel_size": pixel_size,
            "x_fwhm_pixels": x_fwhm / pixel_size,
            "y_fwhm_pixels": y_fwhm / pixel_size,
            "transmission": transmission,
            "aperture_loss": aperture_loss,
            "absorption_loss": absorption_loss,
            "angular_spectrum_loss": 1.0 - captured_power,
            "convolution_boundary_loss": convolution_boundary_loss,
        }
    )
    if (
        not np.isfinite(x_fwhm)
        or not np.isfinite(y_fwhm)
        or min(x_fwhm, y_fwhm) / pixel_size < 4.0
    ):
        stats["sampling_warning"] = (
            "Under-resolved 2D wavefront result: increase Wave points or reduce "
            "Wave window until both FWHM values span at least 4 pixels"
        )
    if angular_aliasing:
        stats["angular_sampling_warning"] = (
            "Wavefront grid does not capture the full source angular spectrum; "
            "reduce the grid spacing"
        )
    return WavefrontResult(
        intensity,
        axis,
        axis,
        detector.z,
        stats,
        transmission,
        {
            "engine": "wavefront",
            "wavelength_m": wavelength,
            "window_m": window_m,
            "points": points,
            "source_model": (
                "point-source angular spectrum plus intensity convolution"
            ),
            "magnification": magnification,
            "point_source_transmission": optical_transmission,
            "captured_angular_power": captured_power,
            "aperture_loss": aperture_loss,
            "absorption_loss": absorption_loss,
            "convolution_boundary_loss": convolution_boundary_loss,
            "focal_position_method": (
                "numerical covariance after explicit aperture clipping"
                if explicit_aperture_clipped
                else "ABCD (lens-aperture diffraction not included)"
            ),
            "focal_positions": focal_positions,
            "boundary_z": envelope_z,
            "boundary_lower_x": envelope_lower["x"],
            "boundary_upper_x": envelope_upper["x"],
            "boundary_lower_y": envelope_lower["y"],
            "boundary_upper_y": envelope_upper["y"],
        },
        profile_extent=(axis[0], axis[-1], axis[0], axis[-1]),
    )


def simulate_wavefront_adaptive(
    source: Source,
    stacks: list[LensStack],
    apertures: list[Aperture],
    detector: Detector,
    energy_keV: float,
    points: int = 512,
    window_m: float = 2e-3,
    minimum_fwhm_pixels: float = 4.0,
) -> WavefrontResult:
    requested_points = points
    requested_window = window_m
    maximum_points = 1048576 if source.dimension == 1 else 2048
    refinements = 0
    window_expansions = 0

    def numeric_stat(result: WavefrontResult, name: str) -> float:
        value = result.stats.get(name, 0.0)
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        return 0.0

    while True:
        result = simulate_wavefront(
            source,
            stacks,
            apertures,
            detector,
            energy_keV,
            points,
            window_m,
        )
        profile = np.asarray(result.profile)
        if profile.ndim == 1:
            edge_power = float(profile[0] + profile[-1])
        else:
            edge_power = float(
                np.sum(profile[0, :])
                + np.sum(profile[-1, :])
                + np.sum(profile[:, 0])
                + np.sum(profile[:, -1])
            )
        edge_fraction = edge_power / max(float(np.sum(profile)), np.finfo(float).tiny)
        if edge_fraction > 1e-6 and points < maximum_points:
            points = min(points * 2, maximum_points)
            window_m *= 2.0
            window_expansions += 1
            continue
        if source.dimension == 1:
            resolved_pixels = numeric_stat(result, "fwhm_pixels")
        else:
            resolved_pixels = min(
                numeric_stat(result, "x_fwhm_pixels"),
                numeric_stat(result, "y_fwhm_pixels"),
            )
        if np.isfinite(resolved_pixels) and resolved_pixels >= minimum_fwhm_pixels:
            result.stats.pop("sampling_warning", None)
            break
        if points >= maximum_points:
            result.stats["sampling_warning"] = (
                f"Wavefront remains under-resolved at the automatic {maximum_points} "
                "point limit; reduce Wave window"
            )
            break
        points = min(points * 2, maximum_points)
        refinements += 1
    result.stats["requested_wave_points"] = requested_points
    result.stats["selected_wave_points"] = points
    result.stats["automatic_refinements"] = refinements
    result.stats["selected_wave_window"] = window_m
    result.stats["automatic_window_expansions"] = window_expansions
    result.metadata["requested_points"] = requested_points
    result.metadata["selected_points"] = points
    result.metadata["automatic_refinements"] = refinements
    result.metadata["requested_window_m"] = requested_window
    result.metadata["selected_window_m"] = window_m
    result.metadata["automatic_window_expansions"] = window_expansions
    return result
