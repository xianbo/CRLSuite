import matplotlib.pyplot as plt

from crlsim.app import CRLSuiteApp


def test_focus_sort_key_is_safe_for_unexpected_labels():
    names = ["Other", "CRL10", "CRL2", "CRL1+CRL3"]
    assert sorted(names, key=CRLSuiteApp._focus_sort_key) == [
        "CRL1+CRL3",
        "CRL2",
        "CRL10",
        "Other",
    ]


def test_report_tables_fit_many_rows():
    figure, axis = plt.subplots()
    rows = [[str(index)] for index in range(100)]
    CRLSuiteApp._draw_report_tables(axis, [("Large", ["Value"], rows)])
    table = axis.tables[0]
    assert min(cell.get_y() for cell in table.get_celld().values()) >= 0
    plt.close(figure)


def test_report_parameter_tables_are_pure():
    data = {
        "mode": "1D",
        "energy": 12.0,
        "source_z": 0.0,
        "spatial_kind": "point",
        "size_x": 0.0,
        "divergence_kind": "collimated",
        "div_x": 0.0,
        "stacks": [
            {
                "location": 2.0,
                "enabled": True,
                "axis": "x",
                "material": "Be",
                "shape": "parabolic",
                "count": 1.0,
                "radius": 0.05,
                "aperture": 0.5,
                "apex": 0.03,
                "pitch": 0.08,
            }
        ],
        "apertures": [],
    }
    original = {**data, "stacks": [dict(data["stacks"][0])]}
    tables = CRLSuiteApp._report_parameter_tables(data)
    assert tables[1][2][0][0] == "CRL1"
    assert data == original
