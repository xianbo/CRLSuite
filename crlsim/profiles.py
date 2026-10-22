import numpy as np
from scipy.optimize import curve_fit


def _arrays(values, weights=None):
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = (
        np.ones_like(values)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if values.shape != weights.shape or np.any(weights < 0):
        raise ValueError("invalid values or weights")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("no positively weighted finite samples")
    return values[valid], weights[valid]


def weighted_profile(values, weights=None, bins=100, range=None):
    values, weights = _arrays(values, weights)
    histogram, edges = np.histogram(values, bins=bins, range=range, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, histogram, edges


def weighted_profile2d(x, y, weights=None, bins=100, range=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = (
        np.ones_like(x)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if x.shape != y.shape or x.shape != weights.shape or np.any(weights < 0):
        raise ValueError("invalid x, y, or weights")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("no positively weighted finite samples")
    x, y, weights = x[valid], y[valid], weights[valid]
    histogram, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=range, weights=weights
    )
    return histogram, xedges, yedges


def interpolated_fwhm(coordinates, values):
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(values, dtype=float)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or len(x) < 2
        or np.any(~np.isfinite(x))
        or np.any(~np.isfinite(y))
    ):
        return np.nan
    baseline = float(np.min(y))
    peak = int(np.argmax(y))
    if y[peak] <= baseline:
        return np.nan
    half = baseline + (y[peak] - baseline) * 0.5
    left_candidates = np.flatnonzero(y[:peak] < half)
    right_candidates = np.flatnonzero(y[peak + 1 :] < half)
    if not len(left_candidates) or not len(right_candidates):
        return np.nan
    left = left_candidates[-1]
    right = peak + 1 + right_candidates[0]
    xl = np.interp(half, [y[left], y[left + 1]], [x[left], x[left + 1]])
    xr = np.interp(half, [y[right], y[right - 1]], [x[right], x[right - 1]])
    return float(xr - xl)


def profile_crop_slice(values, threshold=0.005, padding=0.1):
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or not 0 < threshold < 1 or padding < 0:
        raise ValueError("invalid profile crop inputs")
    marginal = array if array.ndim == 1 else np.sum(array, axis=0)
    marginal = np.maximum(np.nan_to_num(marginal), 0.0)
    peak_value = float(np.max(marginal))
    if peak_value <= 0:
        return slice(0, marginal.size)
    peak = int(np.argmax(marginal))
    smooth_width = max(3, int(np.ceil(marginal.size / 1000)))
    if smooth_width % 2 == 0:
        smooth_width += 1
    kernel = np.ones(smooth_width) / smooth_width
    smooth = np.convolve(marginal, kernel, mode="same")
    cutoff = float(smooth[peak]) * threshold
    first = peak
    last = peak + 1
    while first > 0 and smooth[first - 1] >= cutoff:
        first -= 1
    while last < marginal.size and smooth[last] >= cutoff:
        last += 1
    extra = int(np.ceil((last - first) * padding))
    return slice(max(first - extra, 0), min(last + extra, marginal.size))


def gaussian_fit_fwhm(coordinates, values):
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(values, dtype=float)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or len(x) < 5
        or np.any(~np.isfinite(x))
        or np.any(~np.isfinite(y))
        or np.ptp(y) <= 0
        or np.sum(y) <= 0
    ):
        return np.nan
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))

    def gaussian(v, amplitude, center, width, offset):
        return offset + amplitude * np.exp(-0.5 * ((v - center) / width) ** 2)

    try:
        fit, covariance = curve_fit(
            gaussian,
            x,
            y,
            p0=(
                np.max(y) - np.min(y),
                mean,
                max(sigma, np.finfo(float).eps),
                np.min(y),
            ),
            bounds=([0, -np.inf, 0, -np.inf], np.inf),
            maxfev=10000,
        )
        if np.any(~np.isfinite(fit)) or np.any(~np.isfinite(covariance)):
            return np.nan
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(fit[2]))
    except (RuntimeError, ValueError):
        return np.nan


def statistics(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    bins: int = 100,
    initial_weight: float | None = None,
    aperture_loss: float = 0.0,
    absorption_loss: float = 0.0,
) -> dict[str, object]:
    values, weights = _arrays(values, weights)
    total = float(np.sum(weights))
    centroid = float(np.sum(weights * values) / total)
    sigma = float(np.sqrt(np.sum(weights * (values - centroid) ** 2) / total))
    rms = float(np.sqrt(np.sum(weights * values**2) / total))
    centers, histogram, _ = weighted_profile(values, weights, bins)
    base = np.nan if initial_weight is None else float(initial_weight)
    return {
        "centroid": centroid,
        "sigma": sigma,
        "rms": rms,
        "fwhm": interpolated_fwhm(centers, histogram),
        "gaussian_fwhm": gaussian_fit_fwhm(centers, histogram),
        "weight": total,
        "transmission": total / base if base > 0 else np.nan,
        "aperture_loss": float(aperture_loss),
        "absorption_loss": float(absorption_loss),
    }


def statistics2d(
    position: np.ndarray,
    weights: np.ndarray | None = None,
    initial_weight: float | None = None,
    aperture_loss: float = 0.0,
    absorption_loss: float = 0.0,
    encircled: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, object]:
    position = np.asarray(position, dtype=float)
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must have shape (n, 2)")
    weights = (
        np.ones(len(position)) if weights is None else np.asarray(weights, dtype=float)
    )
    valid = np.all(np.isfinite(position), axis=1) & np.isfinite(weights) & (weights > 0)
    p, w = position[valid], weights[valid]
    if not len(p):
        raise ValueError("no positively weighted finite samples")
    total = np.sum(w)
    centroid = np.sum(p * w[:, None], axis=0) / total
    centered = p - centroid
    covariance = (centered * w[:, None]).T @ centered / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    radii = np.sqrt(np.sum(centered**2, axis=1))
    sorted_order = np.argsort(radii)
    cumulative = np.cumsum(w[sorted_order]) / total
    sorted_radii = radii[sorted_order]
    interpolation_cumulative = np.concatenate(([0.0], cumulative))
    interpolation_radii = np.concatenate(([0.0], sorted_radii))
    enclosed = {}
    for fraction in encircled:
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            enclosed[float(fraction)] = np.nan
        elif fraction > cumulative[-1]:
            enclosed[float(fraction)] = np.nan
        else:
            enclosed[float(fraction)] = float(
                np.interp(fraction, interpolation_cumulative, interpolation_radii)
            )
    base = np.nan if initial_weight is None else float(initial_weight)
    return {
        "centroid": centroid,
        "sigma": np.sqrt(np.diag(covariance)),
        "rms": np.sqrt(np.sum(p**2 * w[:, None], axis=0) / total),
        "covariance": covariance,
        "correlation": float(
            covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        )
        if covariance[0, 0] * covariance[1, 1] > 0
        else np.nan,
        "principal_rms": np.sqrt(np.maximum(eigenvalues[order], 0)),
        "principal_axes": eigenvectors[:, order],
        "encircled_radii": enclosed,
        "weight": float(total),
        "transmission": float(total / base) if base > 0 else np.nan,
        "aperture_loss": float(aperture_loss),
        "absorption_loss": float(absorption_loss),
    }
