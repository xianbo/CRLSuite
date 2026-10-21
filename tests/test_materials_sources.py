import numpy as np
import pytest
import xraylib

from crlsim import DIAMOND, IP_DIP, IP_S, Material, Source


def test_diamond_preset_and_kev_optical_constants():
    assert DIAMOND == Material("C", 3515.0, "diamond")
    assert Material.preset("C") is not DIAMOND
    assert Material.preset("C") == DIAMOND

    low = DIAMOND.optical_constants(10.0)
    high = DIAMOND.optical_constants(20.0)

    assert low.delta > high.delta > 0.0
    assert low.beta > 0.0
    expected_mu = xraylib.CS_Total_CP("C", 10.0) * 3.515 * 100.0
    assert low.mu == pytest.approx(expected_mu)
    assert low.delta == pytest.approx(7.3e-6, rel=0.01)


@pytest.mark.parametrize(
    ("material", "delta_reference", "mu_reference"),
    [(IP_DIP, 1.95e-6, 256.0), (IP_S, 1.87e-6, 246.0)],
)
def test_measured_photoresist_constants(material, delta_reference, mu_reference):
    reference = material.optical_constants(11.867)
    doubled = material.optical_constants(2.0 * 11.867)

    assert reference.delta == pytest.approx(delta_reference)
    assert reference.mu == pytest.approx(mu_reference)
    assert doubled.delta == pytest.approx(delta_reference / 4.0)
    assert doubled.mu == pytest.approx(mu_reference / 8.0)


def test_photoresist_formula_uses_approximation_without_preset_name():
    assert Material("IP-Dip", 1200.0).optical_constants(11.867) == pytest.approx(
        IP_DIP.optical_constants(11.867)
    )


@pytest.mark.parametrize("energy", [0.0, -1.0, np.nan, np.inf])
def test_material_rejects_invalid_kev_energy(energy):
    with pytest.raises(ValueError, match="positive keV"):
        DIAMOND.optical_constants(energy)


def test_source_seed_reproduces_uniform_samples_and_respects_bounds():
    source = Source(
        dimension=2,
        spatial="uniform",
        size=(2.0, 3.0),
        center=(1.0, -1.0),
        spatial_bounds=((0.25, 1.75), (-2.0, 0.5)),
        divergence="uniform",
        angular_size=(0.2, 0.4),
        mean_angle=(0.1, -0.2),
    )

    first = source.sample(2000, seed=1234)
    second = source.sample(2000, seed=1234)
    different = source.sample(2000, seed=1235)

    np.testing.assert_array_equal(first.position, second.position)
    np.testing.assert_array_equal(first.angle, second.angle)
    assert not np.array_equal(first.position, different.position)
    assert np.all((first.position[:, 0] >= 0.25) & (first.position[:, 0] <= 1.75))
    assert np.all((first.position[:, 1] >= -2.0) & (first.position[:, 1] <= 0.5))
    assert np.all((first.angle[:, 0] >= -0.1) & (first.angle[:, 0] <= 0.3))
    assert np.all((first.angle[:, 1] >= -0.6) & (first.angle[:, 1] <= 0.2))


def test_point_and_gaussian_source_distribution_statistics():
    point = Source(
        dimension=2,
        spatial="point",
        center=(2.0, -3.0),
        angular_size=0.0,
        mean_angle=(0.25, -0.5),
    ).sample(4, seed=9)
    np.testing.assert_array_equal(point.position, np.tile((2.0, -3.0), (4, 1)))
    np.testing.assert_array_equal(point.angle, np.tile((0.25, -0.5), (4, 1)))

    gaussian = Source(
        dimension=2,
        spatial="gaussian",
        size=(0.5, 1.5),
        center=(1.0, -2.0),
        divergence="gaussian",
        angular_size=(0.02, 0.04),
        mean_angle=(0.1, -0.1),
    ).sample(20_000, seed=42)
    np.testing.assert_allclose(gaussian.position.mean(axis=0), (1.0, -2.0), atol=0.02)
    np.testing.assert_allclose(gaussian.position.std(axis=0), (0.5, 1.5), rtol=0.02)
    np.testing.assert_allclose(gaussian.angle.mean(axis=0), (0.1, -0.1), atol=0.001)
    np.testing.assert_allclose(gaussian.angle.std(axis=0), (0.02, 0.04), rtol=0.02)


def test_source_rejects_impossible_or_negligible_bounded_support():
    with pytest.raises(ValueError, match="outside bounds"):
        Source(spatial="point", center=2.0, spatial_bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="negligible"):
        Source(
            spatial="gaussian",
            size=1.0,
            center=0.0,
            spatial_bounds=(20.0, 21.0),
        )
