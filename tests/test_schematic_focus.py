from types import SimpleNamespace

import numpy as np

from crlsim.app import CRLSuiteApp


def test_focus_crossing_is_inserted_before_next_optic():
    z, lower, upper = CRLSuiteApp._insert_focus_crossings(
        np.array([0.0, 10.0, 20.0]),
        np.array([0.0, -1.0, -1.0]),
        np.array([0.0, 1.0, 1.0]),
        [{"z_m": 10.0, "focus_m": {"x": 15.0}}],
        "x",
        {"stacks": [{"location": 10.0, "enabled": True}], "apertures": []},
    )

    focus_index = int(np.flatnonzero(z == 15.0)[0])
    assert lower[focus_index] == 0.0
    assert upper[focus_index] == 0.0


def test_focus_crossing_is_not_inserted_after_next_optic():
    z, _, _ = CRLSuiteApp._insert_focus_crossings(
        np.array([0.0, 10.0, 12.0, 20.0]),
        np.array([0.0, -1.0, -0.5, -1.0]),
        np.array([0.0, 1.0, 0.5, 1.0]),
        [{"z_m": 10.0, "focus_m": {"x": 15.0}}],
        "x",
        {
            "stacks": [
                {"location": 10.0, "enabled": True},
                {"location": 12.0, "enabled": True},
            ],
            "apertures": [],
        },
    )

    assert 15.0 not in z


def test_wavefront_envelope_merges_detectors_and_starts_at_point_source():
    app = object.__new__(CRLSuiteApp)
    first = SimpleNamespace(
        detector_z=10.0,
        metadata={
            "boundary_z": [0.0, 10.0],
            "boundary_lower_x": [-1.0, -0.1],
            "boundary_upper_x": [1.0, 0.1],
        },
    )
    second = SimpleNamespace(
        detector_z=20.0,
        metadata={
            "boundary_z": [0.0, 20.0],
            "boundary_lower_x": [-1.0, -0.2],
            "boundary_upper_x": [1.0, 0.2],
        },
    )
    app.detector_results = [first, second]

    z, lower, upper = app._merged_schematic_envelope(
        second,
        "x",
        {"engine": "wavefront", "source_z": 0.0},
    )

    np.testing.assert_allclose(z, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(lower, [0.0, -1e5, -2e5])
    np.testing.assert_allclose(upper, [0.0, 1e5, 2e5])
