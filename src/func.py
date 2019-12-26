import math
import numpy as np
from scipy.ndimage import median_filter


inf = float('inf')


def get_unit(flux):
    return 10**math.floor(math.log10(np.median(flux)))


def select(wave, selectwindow):
    """
    return the indices of wavelength included in the selectwindow
    """
    # print(type(wave))
    # print(wave)
    win = selectwindow[0]
    selected = ((wave > win[0]) & (wave < win[1]))
    for win in selectwindow[1:]:
        selected = (selected | ((wave > win[0]) & (wave < win[1])))
    return np.where(selected)


def mask(wave, maskwindow):
    """
    return the indices of wavelength not included in the selectwindow
    """
    win = maskwindow[0]
    selected = ((wave > win[0]) & (wave < win[1]))
    for win in maskwindow[1:]:
        selected = (selected | ((wave > win[0]) & (wave < win[1])))
    unselected = ~selected
    return np.where(unselected)


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
    arg = np.where(res > 3*std)
    new_flux = flux
    new_flux[arg] = med_flux[arg]
    return new_flux
