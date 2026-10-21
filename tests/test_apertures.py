import numpy as np
import pytest

from crlsim import Aperture, Detector, RayBundle, Simulator
from crlsim.optics import apply_aperture


def test_rectangular_aperture_size_and_offset():
    rays = RayBundle(
        np.array([[0.0, 0.0], [0.6, 0.0], [0.4, 0.6]]),
        np.zeros((3, 2)),
        np.ones(3),
    )
    aperture = Aperture(0.0, "rectangular", 1.0, 1.0, offset_x=0.1)

    loss = apply_aperture(rays, aperture)

    np.testing.assert_array_equal(rays.alive, [True, True, False])
    assert loss == pytest.approx(1.0)


def test_elliptical_aperture_clips_corners():
    rays = RayBundle(
        np.array([[0.0, 0.0], [0.4, 0.4]]),
        np.zeros((2, 2)),
        np.ones(2),
    )

    apply_aperture(rays, Aperture(0.0, "elliptical", 1.0, 1.0))

    np.testing.assert_array_equal(rays.alive, [True, False])


def test_aperture_is_ordered_with_lenses_and_detector():
    rays = RayBundle(np.array([[0.0], [1.0]]), np.zeros((2, 1)), np.ones(2))
    result = Simulator(12.0).run(
        rays,
        [],
        Detector(2.0),
        apertures=[Aperture(1.0, "rectangular", 1.0, 1.0)],
    )

    assert result.trajectory.labels == (
        "source",
        "aperture_0_before",
        "aperture_0",
        "detector",
    )
    assert result.aperture_loss == pytest.approx(1.0)
