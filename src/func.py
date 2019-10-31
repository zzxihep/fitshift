import math
import numpy as np
from scipy.ndimage import median_filter


inf = float('inf')


def get_unit(flux):
    return 10**math.floor(math.log10(np.median(flux)))


def select(wave, flux, err=None, lw=-inf, uw=inf):
    new_wave = np.array(wave)
    new_flux = np.array(flux)
    arg = np.where((new_wave > lw) & (new_wave < uw))
    new_wave = new_wave[arg]
    new_flux = new_flux[arg]
    if err is not None:
        new_err = np.array(err)[arg]
        return new_wave, new_flux, new_err
    return new_wave, new_flux


def mask(wave, maskwindow):
    """
    return the masked arg
    """
    print(type(wave))
    print(wave)
    win = maskwindow[0]
    selected = ((wave > win[0]) & (wave < win[1]))
    for win in maskwindow[1:]:
        selected = (selected | ((wave > win[0]) & (wave < win[1])))
    return np.where(selected)


def trim_edge(wave, flux, err=None, pixelout=(5, 5)):
    il, ir = pixelout
    # print(il, ir)
    new_wave = wave[il:-ir]
    new_flux = flux[il:-ir]
    if err is not None:
        new_err = err[il:-ir]
        return new_wave, new_flux, new_err
    return new_wave, new_flux, None


def median_reject_cos(flux):
    med_flux = median_filter(flux, size=20)
    res = flux - med_flux
    std = np.std(res)
    arg = np.where(res>3*std)
    new_flux = flux
    new_flux[arg] = med_flux[arg]
    return new_flux
