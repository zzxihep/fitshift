import math
import numpy as np


inf = float('inf')


def get_unit(flux):
    return 10**math.floor(math.log10(np.median(flux)))


def select(wave, flux, err=None, lw=-inf, uw=inf):
    new_wave = np.array(wave)
    new_flux = np.array(flux)
    arg = np.where((new_wave > lw) & ( new_wave < uw))
    new_wave = new_wave[arg]
    new_flux = new_flux[arg]
    if err is not None:
        new_err = np.array(err)[arg]
        return new_wave, new_flux, new_err
    return new_wave, new_flux