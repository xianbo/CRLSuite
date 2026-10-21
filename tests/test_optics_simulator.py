from dataclasses import dataclass

import numpy as np
import pytest

from crlsim import Aperture, Detector, LensStack, OpticalConstants, RayBundle, Simulator
from crlsim.optics import apply_lens
from crlsim.simulator import simulate


@dataclass(frozen=True)
class FixedMaterial:
    delta: float = 0.1
    mu: float = 0.0

    def optical_constants(self, energy_keV):
        return OpticalConstants(self.delta, 0.0, self.mu)


def bundle(position, angle=None):
    positions = np.asarray(position, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, None]
    angles = (
        np.zeros_like(positions) if angle is None else np.asarray(angle, dtype=float)
    )
    if angles.ndim == 1:
        angles = angles[:, None]
    return RayBundle(positions, angles, np.ones(len(positions)))


def test_parabolic_lens_changes_angle_and_rays_cross_at_focus():
    rays = bundle([-0.2, 0.2])
    stack = LensStack(0.0, FixedMaterial(), 1, radius=1.0, aperture=2.0)

    focal_distance = 0.2 / np.tan(0.04)
    result = Simulator(10.0).run(rays, [stack], Detector(focal_distance))

    np.testing.assert_allclose(result.trajectory.angle[1, :, 0], (0.04, -0.04))
    np.testing.assert_allclose(result.detector.position[:, 0], 0.0, atol=1e-15)
    assert result.focus["x"] == pytest.approx(focal_distance)


def test_separated_lenses_drift_between_each_stack_element():
    rays = bundle([-0.2, 0.2])
    stack = LensStack(
        2.0,
        FixedMaterial(),
        2,
        radius=1.0,
        aperture=2.0,
        pitch=2.0,
    )

    result = Simulator(10.0).run(rays, [stack], Detector(4.0))

    assert result.trajectory.labels == (
        "source",
        "stack_0_lens_0",
        "stack_0_lens_1",
        "detector",
    )
    second_position = 0.2 - 2.0 * np.tan(0.04)
    second_angle = 0.04 + 0.2 * second_position
    detector_position = second_position - np.tan(second_angle)
    np.testing.assert_allclose(
        result.trajectory.position[2, :, 0],
        (-second_position, second_position),
    )
    np.testing.assert_allclose(
        result.trajectory.angle[2, :, 0],
        (second_angle, -second_angle),
    )
    np.testing.assert_allclose(
        result.detector.position[:, 0],
        (-detector_position, detector_position),
    )


def test_detector_stops_propagation_between_lenses():
    rays = bundle([-0.2, 0.2])
    stack = LensStack(
        2.0,
        FixedMaterial(),
        2,
        radius=1.0,
        aperture=2.0,
        pitch=2.0,
    )

    result = Simulator(10.0).run(rays, [stack], Detector(2.0))

    assert result.trajectory.z.tolist() == [0.0, 1.0, 2.0]
    assert result.trajectory.labels[2] == "detector"
    detector_position = 0.2 - np.tan(0.04)
    np.testing.assert_allclose(
        result.detector.position[:, 0],
        (-detector_position, detector_position),
    )
    np.testing.assert_allclose(result.detector.angle[:, 0], (0.04, -0.04))


def test_detector_follows_colocated_optics_and_wrapper_forwards_planes():
    rays = bundle([0.0, 0.4])
    stack = LensStack(2.0, FixedMaterial(), 1, radius=1.0, aperture=2.0)
    aperture = Aperture(1.0, "rectangular", 0.5, 1.0)

    result = simulate(
        rays,
        [stack],
        Detector(2.0),
        10.0,
        source_plane=-1.0,
        apertures=[aperture],
    )

    assert result.trajectory.z.tolist() == [-1.0, 1.0, 1.0, 2.0, 2.0]
    assert result.trajectory.labels[-2:] == ("stack_0_lens_0", "detector")
    np.testing.assert_array_equal(result.detector.alive, [True, False])


def test_aperture_clipping_and_absorption_losses_are_separate():
    rays = bundle([0.0, 0.6])
    stack = LensStack(
        0.0,
        FixedMaterial(mu=2.0),
        1,
        radius=1.0,
        aperture=1.0,
        apex_thickness=0.5,
    )

    aperture_loss, absorption_loss = apply_lens(rays, stack, delta=0.1, mu=2.0)

    np.testing.assert_array_equal(rays.alive, (True, False))
    np.testing.assert_allclose(rays.weight, (np.exp(-1.0), 0.0))
    assert aperture_loss == pytest.approx(1.0)
    assert absorption_loss == pytest.approx(1.0 - np.exp(-1.0))


def test_detector_aperture_is_included_in_reported_loss():
    rays = bundle([0.0, 0.4])
    result = Simulator(10.0).run(rays, [], Detector(0.0, aperture=0.5))

    np.testing.assert_array_equal(result.detector.alive, (True, False))
    assert result.aperture_loss == pytest.approx(1.0)
    assert result.absorption_loss == pytest.approx(0.0)


def test_half_lens_has_single_surface_power_and_sag_absorption():
    full_rays = bundle([0.2])
    half_rays = bundle([0.2])
    full = LensStack(0.0, FixedMaterial(), 1.0, 1.0, 1.0, 0.1)
    half = LensStack(0.0, FixedMaterial(), 0.5, 1.0, 1.0, 0.1)

    apply_lens(full_rays, full, delta=0.1, mu=2.0)
    apply_lens(half_rays, half, delta=0.1, mu=2.0, surface_factor=0.5)

    assert half_rays.angle[0, 0] == pytest.approx(full_rays.angle[0, 0] / 2.0)
    assert half_rays.weight[0] == pytest.approx(np.exp(-2.0 * (0.1 + 0.02)))
    assert full_rays.weight[0] == pytest.approx(np.exp(-2.0 * (0.1 + 0.04)))


def test_half_integer_stack_centers_and_factors():
    stack = LensStack(1.0, FixedMaterial(), 2.5, 1.0, 1.0, pitch=0.2)

    assert stack.centers == pytest.approx([0.8, 1.0, 1.2])
    assert [stack.surface_factor(index) for index in range(3)] == [1.0, 1.0, 0.5]


def test_explicit_bundle_can_start_at_offset_source_plane():
    rays = bundle([-0.1, 0.1], [-0.01, 0.01])

    result = Simulator(10.0).run(rays, [], Detector(1.0), source_plane=-2.0)

    assert result.trajectory.z.tolist() == [-2.0, 1.0]
    np.testing.assert_allclose(
        result.detector.position[:, 0],
        [-0.1 - 3.0 * np.tan(0.01), 0.1 + 3.0 * np.tan(0.01)],
    )


def test_reports_absolute_focus_after_each_stack():
    rays = bundle([-0.2, 0.2])
    stack = LensStack(1.0, FixedMaterial(), 2.0, 1.0, 2.0, pitch=1.0)

    result = Simulator(10.0).run(rays, [stack], Detector(3.0))
    focal_positions = result.metadata["focal_positions"]

    assert [item["label"] for item in focal_positions] == ["stack_0"]
    assert focal_positions[0]["z_m"] == pytest.approx(1.5)
    assert np.isfinite(focal_positions[0]["focus_m"]["x"])
