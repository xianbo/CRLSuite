# CRLSuite

CRLSuite is a Python 3.13 desktop application for forward simulation of
compound refractive X-ray lenses, apertures, and multiple detector planes.

## Features

- Switchable geometric ray tracing and coherent Fresnel wavefront propagation.
- 1D and 2D sources with Gaussian, uniform, or point spatial distributions.
- Multiple separated CRL stacks, rectangular/elliptical apertures, and detectors.
- Xraylib refraction and total attenuation, including apex and sag absorption.
- Built-in IP-Dip and IP-S photoresist models. Their supplied delta values scale
  as energy⁻²; measured attenuation anchors of 256 and 246 m⁻¹ at 11.867 keV
  respectively are approximated as energy⁻³ away from absorption edges.
- Dynamic X/Y beamline schematics, profiles, statistics, JSON configurations,
  HDF5 results, and PNG reports.

## Units

Inputs and outputs use SI units unless an API explicitly states otherwise:
lengths are metres, angles are radians, and photon energies are keV.
Material density inputs follow the units required by `xraylib` and should be
converted at that boundary rather than treated as SI density values.

## Installation

Install Python 3.13 or newer, extract the ZIP, open a terminal in the extracted
`CRLSuite` folder, then run:

```bash
python -m pip install .
crlsuite
```

On Linux, install your distribution's Tk package (commonly `python3-tk`) if
Python was built without Tkinter.

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Quick use

1. Select 1D/2D and ray-tracing/wavefront mode.
2. Define the source and photon energy.
3. Add CRLs and apertures in the **Optics** panel.
4. Add one or more detectors and click **Calculate**.
5. Use the `D1`, `D2`, ... buttons to select the displayed detector profile.
6. Use **Save**/**Load** for JSON configurations, **Save Results** for HDF5,
   and **Generate Report** for a PNG summary.

Example configurations and reports are included in `examples/`.

## Development

```bash
pytest
ruff check .
mypy crlsim
```
