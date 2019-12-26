from lmfit import Parameters, minimize
from lmfit import report_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
import convol
import template
import specio


def residual(pars, x, data, eps=None):
    realpar = [t.value for t in pars.values()]
    res = data - np.array(convol.legendre_poly(x, realpar))
    if eps is not None:
        return res / eps
    return res


def get_scale(wave, pars):
    """
    return numpy.ndarray
    """
    realpar = [t.value for t in pars.values()]
    return np.array(convol.legendre_poly(wave, realpar))


def continuum2(wave, flux, err=None):
    newflux = median_filter(flux, 60)
    plt.plot(wave, flux)
    plt.plot(wave, newflux)
    plt.show()


def continuum(wave, flux, err=None):
    print(len(wave))
    print(len(flux))
    print(wave)
    pars = Parameters()
    template.set_pars(pars, 'scale', 6, [0.0]*7)
    result = minimize(residual, pars, args=(wave, flux, err))
    scale = get_scale(wave, result.params)
    tmpuniform = flux / scale
    std = np.std(tmpuniform)
    arg = np.where(tmpuniform > 1-3*std)
    new_wave = wave[arg]
    new_flux = flux[arg]
    if err is not None:
        new_err = err[arg]
    else:
        new_err = None
    size = len(wave)
    # plt.plot(wave, flux)
    # plt.plot(wave, scale)
    # plt.show()
    while size != len(new_wave):
        pars = result.params
        result = minimize(residual, pars, args=(new_wave, new_flux, new_err))
        scale = get_scale(new_wave, result.params)
        tmpwave = new_wave
        tmpuniform = new_flux / scale
        std = np.std(tmpuniform)
        print('std = ')
        print(std)
        print(1-3*std)
        arg = np.where(tmpuniform > 1 - 3*std)
        size = len(new_wave)
        new_wave = new_wave[arg]
        new_flux = new_flux[arg]
        if err is not None:
            new_err = new_err[arg]
        # plt.plot(wave, flux)
        # plt.plot(tmpwave, scale)
        # plt.figure()
        # plt.plot(tmpwave, tmpuniform)
        # plt.show()
    report_fit(result)
    return result


def uniform(wave, flux, err=None, show=False):
    length = wave[-1] - wave[0]
    center = (wave[0] + wave[-1])/2
    new_x = (wave-center)/length*1.99
    result = continuum(new_x, flux, err)
    scale = get_scale(new_x, result.params)
    new_flux = flux / scale
    if show is True:
        plt.plot(wave, flux)
        plt.plot(wave, scale)
        plt.figure()
        plt.plot(wave, new_flux)
        plt.show()
    if err is not None:
        return new_flux, err / scale
    return new_flux


def test():
    fname = "data/spec-4961-55719-0378.fits"
    wave, flux, err = specio.read_sdss(fname)
    flux = flux * 1.0e15
    err = err * 1.0e15
    length = wave[-1] - wave[0]
    center = (wave[0] + wave[-1])/2
    new_x = (wave-center)/length*1.99
    continuum2(new_x, flux, err)
    result = continuum(new_x, flux, err)
    scale = get_scale(new_x, result.params)
    cont = flux / scale
    plt.plot(wave, flux)
    plt.plot(wave, scale)
    plt.figure()
    plt.plot(wave, cont)
    plt.show()


if __name__ == '__main__':
    test()
