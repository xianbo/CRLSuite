import json

import h5py
import numpy as np

from crlsim import Detector, RayBundle, Simulator
from crlsim.export import save_hdf5, save_hdf5_multiple
from crlsim.wavefront import WavefrontResult


def test_hdf5_export_contains_configuration_and_results(tmp_path):
    rays = RayBundle(
        np.array([[-1e-6], [1e-6]]),
        np.zeros((2, 1)),
        np.ones(2),
    )
    result = Simulator(12.0).run(rays, [], Detector(1.0))
    result.profile = np.array([1.0, 2.0])
    result.profile_coordinates = np.array([-1e-6, 1e-6])
    result.stats = {"transmission": 1.0}
    path = tmp_path / "results.h5"

    save_hdf5(
        path,
        {"mode": "1D", "stacks": [], "points": np.int64(2)},
        result,
    )

    with h5py.File(path, "r") as handle:
        assert handle.attrs["format"] == "CRLSuite results"
        assert json.loads(handle.attrs["configuration_json"])["points"] == 2
        assert handle["configuration"].attrs["mode"] == "1D"
        assert "source_rays" not in handle
        assert "detector_rays" not in handle
        assert "trajectory" not in handle
        np.testing.assert_allclose(handle["detector_profile"][:], [1.0, 2.0])
        assert handle["statistics/transmission"][()] == 1.0


def test_multi_detector_hdf5_export(tmp_path):
    rays = RayBundle(np.array([[0.0]]), np.zeros((1, 1)), np.ones(1))
    first = Simulator(12.0).run(rays, [], Detector(1.0))
    second = Simulator(12.0).run(rays, [], Detector(2.0))
    first.profile = second.profile = np.ones(3)
    first.stats = second.stats = {"fwhm": 1e-6}
    path = tmp_path / "multiple.h5"

    save_hdf5_multiple(path, {"detectors": [{}, {}]}, [first, second])

    with h5py.File(path, "r") as handle:
        assert handle["detectors/1"].attrs["z_m"] == 1.0
        assert handle["detectors/2"].attrs["z_m"] == 2.0
        np.testing.assert_allclose(handle["detectors/2/profile"][:], 1.0)
        assert "detectors/1/envelope" in handle
        assert "detectors/1/focus" in handle
        assert "detectors/1/metadata" in handle


def test_wavefront_hdf5_export_contains_coordinates_and_transmission(tmp_path):
    coordinates = np.array([-1e-6, 0.0, 1e-6])
    result = WavefrontResult(
        profile=np.ones((3, 3)),
        coordinates_x=coordinates,
        coordinates_y=coordinates,
        detector_z=1.0,
        stats={"peak": np.float64(1.0)},
        transmission=0.75,
        metadata={"samples": np.int64(3), "bounds": coordinates},
        profile_extent=(-1e-6, 1e-6, -1e-6, 1e-6),
    )
    path = tmp_path / "wavefront.h5"

    save_hdf5(path, {"window": coordinates}, result)

    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["coordinates_x_m"][:], coordinates)
        np.testing.assert_array_equal(handle["coordinates_y_m"][:], coordinates)
        assert handle.attrs["transmission"] == 0.75
        assert handle["metadata/samples"][()] == 3
        assert json.loads(handle.attrs["configuration_json"])["window"] == [
            -1e-6,
            0.0,
            1e-6,
        ]
