import numpy as np
import convol

c = 299792.458


def gaussian_filter(wave, flux, fwhm=None, R=None):
    if fwhm is not None:
        sigma = fwhm / 2.355
        k0 = sigma / c
    else:
        k0 = 1. / (2.355*R)
    return np.array(convol.gauss_filter(wave, flux, [0.0, k0]))
