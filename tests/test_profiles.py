import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeWarning

from crlsim import (
    gaussian_fit_fwhm,
    interpolated_fwhm,
    statistics,
    statistics2d,
    weighted_profile2d,
)
from crlsim.profiles import profile_crop_slice


def test_gaussian_profile_fit_and_weighted_statistics():
    coordinates = np.linspace(-5.0, 5.0, 401)
    sigma = 1.25
    profile = 3.0 + 7.0 * np.exp(-0.5 * ((coordinates - 0.4) / sigma) ** 2)
    expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        fitted_fwhm = gaussian_fit_fwhm(coordinates, profile)
    assert fitted_fwhm == pytest.approx(expected_fwhm, rel=1e-6)

    result = statistics(
        [-1.0, 1.0, 3.0],
        weights=[1.0, 2.0, 1.0],
        bins=3,
        initial_weight=8.0,
        aperture_loss=2.0,
        absorption_loss=2.0,
    )
    assert result["centroid"] == pytest.approx(1.0)
    assert result["sigma"] == pytest.approx(np.sqrt(2.0))
    assert result["rms"] == pytest.approx(np.sqrt(3.0))
    assert result["weight"] == pytest.approx(4.0)
    assert result["transmission"] == pytest.approx(0.5)
    assert result["aperture_loss"] == pytest.approx(2.0)
    assert result["absorption_loss"] == pytest.approx(2.0)


def test_weighted_2d_profile_preserves_total_positive_weight():
    histogram, x_edges, y_edges = weighted_profile2d(
        [0.0, 0.0, 1.0, np.nan],
        [0.0, 1.0, 1.0, 0.0],
        weights=[1.0, 2.0, 3.0, 100.0],
        bins=(2, 2),
        range=((-0.5, 1.5), (-0.5, 1.5)),
    )

    np.testing.assert_array_equal(histogram, ((1.0, 2.0), (0.0, 3.0)))
    np.testing.assert_allclose(x_edges, (-0.5, 0.5, 1.5))
    np.testing.assert_allclose(y_edges, (-0.5, 0.5, 1.5))
    assert histogram.sum() == pytest.approx(6.0)


def test_statistics2d_weighted_covariance_axes_and_encircled_radii():
    positions = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0], [9.0, 9.0]])
    result = statistics2d(
        positions,
        weights=[1.0, 2.0, 1.0, 0.0],
        initial_weight=8.0,
        encircled=(0.5, 1.0),
    )

    np.testing.assert_allclose(result["centroid"], (0.0, 0.0))
    np.testing.assert_allclose(result["sigma"], (np.sqrt(2.0), 0.0))
    np.testing.assert_allclose(result["rms"], (np.sqrt(2.0), 0.0))
    np.testing.assert_allclose(result["covariance"], ((2.0, 0.0), (0.0, 0.0)))
    np.testing.assert_allclose(result["principal_rms"], (np.sqrt(2.0), 0.0))
    assert np.isnan(result["correlation"])
    assert result["encircled_radii"][0.5] == pytest.approx(0.0)
    assert result["encircled_radii"][1.0] == pytest.approx(2.0)
    assert result["weight"] == pytest.approx(4.0)
    assert result["transmission"] == pytest.approx(0.5)


def test_profile_edge_cases_are_explicit_and_clipped():
    assert interpolated_fwhm(
        np.arange(7.0), [2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 2.0]
    ) == pytest.approx(2.0)
    assert profile_crop_slice([0.0, 0.0, 1.0, 0.0, 0.0], padding=0) == slice(1, 4)
    assert np.isnan(gaussian_fit_fwhm(np.arange(4.0), np.ones(4)))
    assert np.isnan(statistics([0.0, 1.0])["transmission"])
    result = statistics2d(np.array([[0.0, 0.0], [2.0, 0.0]]), encircled=(-1.0, 2.0))
    assert np.isnan(result["encircled_radii"][-1.0])
    assert np.isnan(result["encircled_radii"][2.0])
    assert np.isnan(result["transmission"])
