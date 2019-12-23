import convol


def gaussian_filter(wave, flux, fwhm=None, R=None):
    sigma = fwhm / 2.355
