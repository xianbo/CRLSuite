import numpy as np
import pytest

from crlsim import Detector, LensStack, OpticalConstants, RayBundle, Simulator
from crlsim.optics import apply_lens


class FixedMaterial:
    def optical_constants(self, energy_keV):
        return OpticalConstants(0.1, 0.0, 0.0)


def test_cylindrical_lens_focuses_only_selected_axis():
    rays = RayBundle(
        position=np.array([[-0.2, -0.3], [0.2, 0.3]]),
        angle=np.array([[0.01, 0.02], [0.01, 0.02]]),
        weight=np.ones(2),
    )
    stack = LensStack(
        0.0,
        FixedMaterial(),
        1,
        radius=1.0,
        aperture=2.0,
        shape="parabolic",
        axis="x",
    )

    apply_lens(rays, stack, delta=0.1, mu=0.0)

    np.testing.assert_allclose(rays.angle[:, 0], (0.05, -0.03))
    np.testing.assert_allclose(rays.angle[:, 1], (0.02, 0.02))


def test_directional_lens_does_not_clip_flat_axis():
    rays = RayBundle(
        position=np.array([[0.0, -5.0], [0.0, 5.0]]),
        angle=np.zeros((2, 2)),
        weight=np.ones(2),
    )
    stack = LensStack(
        0.0,
        FixedMaterial(),
        1,
        radius=1.0,
        aperture=1.0,
        shape="parabolic",
        axis="x",
    )

    apply_lens(rays, stack, delta=0.1, mu=0.0)

    np.testing.assert_array_equal(rays.alive, [True, True])


def test_directional_lens_acts_as_x_lens_in_one_dimension():
    rays = RayBundle(np.array([[-0.1], [0.1]]), np.zeros((2, 1)), np.ones(2))
    stack = LensStack(
        0.0,
        FixedMaterial(),
        1,
        radius=1.0,
        aperture=1.0,
        shape="parabolic",
        axis="x",
    )

    result = Simulator(10.0).run(rays, [stack], Detector(1.0))
    assert np.any(result.detector.angle[:, 0] != 0)


def test_spherical_lens_validates_aperture_against_diameter():
    with pytest.raises(ValueError, match="95% of diameter"):
        LensStack(
            0.0,
            FixedMaterial(),
            1,
            radius=1.0,
            aperture=2.0,
            shape="spherical",
        )


def test_spherical_lens_uses_sag_and_local_surface_radius():
    rays = RayBundle(np.array([[0.6]]), np.zeros((1, 1)), np.ones(1))
    stack = LensStack(
        0.0,
        FixedMaterial(),
        1,
        radius=1.0,
        aperture=1.9,
        apex_thickness=0.1,
        shape="spherical",
    )

    _, absorption_loss = apply_lens(rays, stack, delta=0.1, mu=2.0)

    expected_thickness = 0.1 + 2.0 * (1.0 - 0.8)
    assert rays.angle[0, 0] == pytest.approx(-0.15)
    assert rays.weight[0] == pytest.approx(np.exp(-2.0 * expected_thickness))
    assert absorption_loss == pytest.approx(1.0 - rays.weight[0])


@pytest.mark.parametrize("axis", ["z", "X"])
def test_lens_requires_valid_focus_axis(axis):
    with pytest.raises(ValueError, match="focus axis"):
        LensStack(
            0.0,
            FixedMaterial(),
            1,
            radius=1.0,
            aperture=1.0,
            shape="parabolic",
            axis=axis,
        )
