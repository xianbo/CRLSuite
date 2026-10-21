import numpy as np
import pytest

from crlsim import BERYLLIUM, DIAMOND, Aperture, Detector, LensStack, Source
from crlsim.wavefront import (
    _phase_gradient,
    simulate_wavefront,
    simulate_wavefront_adaptive,
)


def test_wavefront_free_space_preserves_power():
    source = Source(dimension=1, spatial="gaussian", size=20e-6)

    result = simulate_wavefront(
        source, [], [], Detector(1.0), 12.0, points=256, window_m=1e-3
    )

    assert result.profile.shape == (256,)
    assert result.transmission + result.stats["convolution_boundary_loss"] == (
        pytest.approx(1.0, rel=1e-10)
    )
    assert result.stats["wave_pixel_size"] == pytest.approx(1e-3 / 256)


def test_wavefront_aperture_reduces_transmission():
    source = Source(dimension=1, spatial="gaussian", size=100e-6)

    result = simulate_wavefront(
        source,
        [],
        [Aperture(0.0, "rectangular", 50e-6, 1e-3)],
        Detector(0.1),
        12.0,
        points=256,
        window_m=1e-3,
    )

    assert 0.0 < result.transmission < 1.0
    assert result.metadata["boundary_z"] == [0.0, 0.0, 0.0, 0.1]
    slit_index = 2
    pixel = 1e-3 / 256
    assert result.metadata["boundary_upper_x"][slit_index] <= 25e-6 + 3 * pixel
    assert result.metadata["boundary_lower_x"][slit_index] >= -25e-6 - 3 * pixel
    assert "wavefront_divergence_x" in result.stats
    assert "wavefront_divergence_fwhm_x" in result.stats
    assert "wavefront_divergence_gaussian_fwhm_x" in result.stats
    assert "wavefront_virtual_source_x" in result.stats


def test_wavefront_2d_lens_produces_detector_image():
    source = Source(dimension=2, spatial="gaussian", size=(50e-6, 50e-6))
    lens = LensStack(0.1, BERYLLIUM, 1.0, 100e-6, 500e-6, 30e-6)

    result = simulate_wavefront(
        source, [lens], [], Detector(0.2), 12.0, points=64, window_m=1e-3
    )

    assert result.profile.shape == (64, 64)
    assert np.all(np.isfinite(result.profile))
    assert "x_fwhm" in result.stats
    assert "y_fwhm" in result.stats
    assert "x_gaussian_fwhm" in result.stats
    assert "y_gaussian_fwhm" in result.stats
    focal_positions = result.metadata["focal_positions"]
    assert [item["label"] for item in focal_positions] == ["stack_0"]
    assert np.isfinite(focal_positions[0]["focus_m"]["x"])
    assert np.isfinite(focal_positions[0]["focus_m"]["y"])


def test_wavefront_stack_focus_uses_point_source_conjugate():
    source = Source(
        z=0.0,
        dimension=1,
        spatial="point",
        divergence="gaussian",
        angular_size=10e-6,
    )
    lens = LensStack(10.0, BERYLLIUM, 1.0, 100e-6, 500e-6, 30e-6)
    delta = BERYLLIUM.optical_constants(12.0).delta
    focal_length = lens.radius / (2.0 * delta)
    expected_focus = lens.z + 1.0 / (1.0 / focal_length - 1.0 / lens.z)

    result = simulate_wavefront(
        source, [lens], [], Detector(12.0), 12.0, points=4096, window_m=1e-3
    )

    focus = result.metadata["focal_positions"][0]["focus_m"]["x"]
    assert focus == pytest.approx(expected_focus, rel=0.02)


def test_strong_stack_focus_uses_stable_conjugate_without_apertures():
    source = Source(
        z=0.0,
        dimension=1,
        spatial="point",
        divergence="gaussian",
        angular_size=4e-6,
    )
    lens = LensStack(53.432, DIAMOND, 10.0, 50e-6, 400e-6, 30e-6, 1.2e-3)

    result = simulate_wavefront(
        source, [lens], [], Detector(54.987), 20.98, points=2048, window_m=2e-3
    )

    focus = result.metadata["focal_positions"][0]["focus_m"]["x"]
    assert focus == pytest.approx(54.987, abs=0.002)


def test_upstream_slit_changes_numerical_stack_focus():
    source = Source(
        z=0.0,
        dimension=1,
        spatial="point",
        divergence="gaussian",
        angular_size=20e-6,
    )
    lens = LensStack(10.0, BERYLLIUM, 1.0, 100e-6, 500e-6, 30e-6)
    open_result = simulate_wavefront(
        source, [lens], [], Detector(12.0), 12.0, points=4096, window_m=1e-3
    )
    slit_result = simulate_wavefront(
        source,
        [lens],
        [Aperture(5.0, "rectangular", 20e-6, 1e-3)],
        Detector(12.0),
        12.0,
        points=4096,
        window_m=1e-3,
    )

    open_focus = open_result.metadata["focal_positions"][0]["focus_m"]["x"]
    slit_focus = slit_result.metadata["focal_positions"][0]["focus_m"]["x"]
    assert abs(open_focus - slit_focus) > 1e-5


def test_phase_gradient_resolves_rapid_quadratic_phase():
    points = 2048
    spacing = 1e-6
    x = (np.arange(points) - points // 2) * spacing
    coefficient = 1.0e9
    field = np.exp(1j * coefficient * x**2)

    gradient = _phase_gradient(field, spacing, 0)
    expected = 2.0 * coefficient * x
    central = np.abs(x) < 0.5e-3

    np.testing.assert_allclose(
        gradient[central], expected[central], rtol=1e-9, atol=1e-6
    )


def test_wavefront_2d_rejects_excessive_grid():
    source = Source(dimension=2, spatial="gaussian", size=(50e-6, 50e-6))

    with pytest.raises(ValueError, match="cannot exceed 2048"):
        simulate_wavefront(
            source, [], [], Detector(0.2), 12.0, points=4096, window_m=1e-3
        )


def test_finite_source_is_convolved_with_numerical_magnification():
    point = Source(
        dimension=1,
        spatial="point",
        divergence="gaussian",
        angular_size=20e-6,
    )
    extended = Source(
        dimension=1,
        spatial="gaussian",
        size=30e-6,
        divergence="gaussian",
        angular_size=20e-6,
    )

    point_result = simulate_wavefront(
        point, [], [], Detector(1.0), 12.0, points=1024, window_m=2e-3
    )
    extended_result = simulate_wavefront(
        extended, [], [], Detector(1.0), 12.0, points=1024, window_m=2e-3
    )

    assert extended_result.stats["sigma"] > point_result.stats["sigma"]
    assert extended_result.metadata["magnification"] == pytest.approx([1.0])
    assert "intensity convolution" in extended_result.metadata["source_model"]


def test_native_grid_gaussian_fwhm_matches_gaussian_fit():
    source = Source(
        dimension=1,
        spatial="gaussian",
        size=30e-6,
        divergence="gaussian",
        angular_size=1e-6,
    )

    result = simulate_wavefront(
        source, [], [], Detector(0.1), 12.0, points=4096, window_m=1e-3
    )

    assert result.stats["fwhm"] == pytest.approx(
        result.stats["gaussian_fwhm"], rel=0.03
    )


def test_adaptive_sampling_increases_underresolved_1d_grid():
    source = Source(
        dimension=1,
        spatial="point",
        divergence="gaussian",
        angular_size=2e-6,
    )

    result = simulate_wavefront_adaptive(
        source, [], [], Detector(1.0), 12.0, points=32, window_m=1e-3
    )

    assert result.stats["selected_wave_points"] > 32
    assert result.stats["automatic_refinements"] > 0
    assert result.metadata["selected_points"] == result.stats["selected_wave_points"]


def test_adaptive_sampling_expands_clipped_window():
    source = Source(
        dimension=1,
        spatial="point",
        divergence="uniform",
        angular_size=100e-6,
    )

    result = simulate_wavefront_adaptive(
        source, [], [], Detector(2.0), 12.0, points=128, window_m=50e-6
    )

    assert result.stats["automatic_window_expansions"] > 0
    assert result.stats["selected_wave_window"] > 50e-6


def test_oversized_aperture_is_numerically_transparent():
    source = Source(
        dimension=1,
        spatial="point",
        divergence="gaussian",
        angular_size=2e-6,
    )
    without = simulate_wavefront(
        source, [], [], Detector(0.1), 12.0, points=512, window_m=1e-3
    )
    with_large = simulate_wavefront(
        source,
        [],
        [Aperture(0.05, "rectangular", 0.9e-3, 1e-3)],
        Detector(0.1),
        12.0,
        points=512,
        window_m=1e-3,
    )

    np.testing.assert_allclose(
        with_large.profile, without.profile, rtol=1e-10, atol=1e-12
    )


def test_narrow_slit_increases_wavefront_divergence():
    source = Source(
        dimension=1,
        spatial="point",
        divergence="gaussian",
        angular_size=5e-6,
    )
    open_result = simulate_wavefront(
        source, [], [], Detector(1.0), 12.0, points=2048, window_m=1e-3
    )
    slit_result = simulate_wavefront(
        source,
        [],
        [Aperture(0.5, "rectangular", 2e-6, 1e-3)],
        Detector(1.0),
        12.0,
        points=2048,
        window_m=1e-3,
    )

    assert (
        slit_result.stats["wavefront_divergence_x"]
        > open_result.stats["wavefront_divergence_x"]
    )


def test_detector_aperture_is_applied_and_reported_as_aperture_loss():
    source = Source(dimension=1, angular_size=20e-6, divergence="gaussian")
    result = simulate_wavefront(
        source, [], [], Detector(1.0, aperture=20e-6), 12.0, 2048, 1e-3
    )

    assert 0.0 < result.transmission < 1.0
    assert result.stats["aperture_loss"] == pytest.approx(1.0 - result.transmission)
    assert result.stats["absorption_loss"] == pytest.approx(0.0)
    assert np.all(result.profile[np.abs(result.coordinates_x) > 10e-6] == 0.0)


def test_lens_losses_are_separated_and_conserved():
    source = Source(dimension=1, angular_size=20e-6, divergence="gaussian")
    lens = LensStack(0.2, BERYLLIUM, 1.0, 100e-6, 40e-6, 30e-6)
    result = simulate_wavefront(source, [lens], [], Detector(0.3), 12.0, 2048, 1e-3)

    accounted = (
        result.transmission
        + result.stats["aperture_loss"]
        + result.stats["absorption_loss"]
        + result.stats["convolution_boundary_loss"]
        + result.stats["angular_spectrum_loss"]
    )
    assert result.stats["aperture_loss"] > 0.0
    assert result.stats["absorption_loss"] > 0.0
    assert accounted == pytest.approx(1.0, abs=1e-12)


def test_source_center_shifts_point_and_extended_source_without_pixel_bias():
    pixel = 1e-3 / 1024
    point = Source(
        dimension=1,
        spatial="point",
        center=20 * pixel,
        divergence="gaussian",
        angular_size=20e-6,
    )
    extended = Source(
        dimension=1,
        spatial="gaussian",
        size=8e-6,
        center=20 * pixel,
        divergence="gaussian",
        angular_size=20e-6,
    )

    point_result = simulate_wavefront(point, [], [], Detector(0.0), 12.0, 1024, 1e-3)
    extended_result = simulate_wavefront(
        extended, [], [], Detector(0.0), 12.0, 1024, 1e-3
    )

    assert point_result.stats["centroid"] == pytest.approx(20 * pixel, abs=pixel / 20)
    assert extended_result.stats["centroid"] == pytest.approx(
        20 * pixel, abs=pixel / 20
    )


def test_angular_nyquist_loss_is_not_normalized_away():
    source = Source(dimension=1, divergence="uniform", angular_size=1e-3)
    result = simulate_wavefront(source, [], [], Detector(0.0), 12.0, 256, 1e-3)

    assert result.transmission < 0.1
    assert result.metadata["captured_angular_power"] == pytest.approx(
        result.transmission
    )
    assert "angular_sampling_warning" in result.stats


def test_axis_y_cylindrical_lens_is_safe_for_1d_wavefront():
    source = Source(dimension=1, angular_size=5e-6, divergence="gaussian")
    lens = LensStack(0.1, BERYLLIUM, 1.0, 100e-6, 500e-6, 30e-6, axis="y")

    result = simulate_wavefront(source, [lens], [], Detector(0.2), 12.0, 512, 1e-3)
    assert np.all(np.isfinite(result.profile))
