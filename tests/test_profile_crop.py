import numpy as np

from crlsim.profiles import profile_crop_slice


def test_profile_crop_reduces_empty_1d_window():
    profile = np.zeros(1000)
    profile[490:510] = 1.0

    cropped = profile_crop_slice(profile)

    assert cropped.start > 400
    assert cropped.stop < 600
    assert np.sum(profile[cropped]) == np.sum(profile)


def test_profile_crop_uses_2d_marginal():
    image = np.zeros((100, 200))
    image[45:55, 90:110] = 1.0

    x_crop = profile_crop_slice(image)
    y_crop = profile_crop_slice(image.T)

    assert x_crop.start > 50 and x_crop.stop < 150
    assert y_crop.start > 20 and y_crop.stop < 80


def test_profile_crop_excludes_disconnected_diffraction_orders():
    profile = np.zeros(501)
    profile[240:261] = np.hanning(21)
    profile[180:191] = 0.05 * np.hanning(11)
    profile[310:321] = 0.05 * np.hanning(11)

    cropped = profile_crop_slice(profile)

    assert cropped.start > 190
    assert cropped.stop < 310
