import numpy as np
from astropy.io import fits
import func


class Spectrum:
    def __init__(self, wave, flux, err=None, filename=None):
        self.wave = wave
        self.flux = flux
        self.err = err
        self.unit = func.get_unit(self.flux)
        self.filename = filename

    def __init__(self, fn):
        self.filename = fn
        self.wave, self.fux, self.err = read_iraf(fn)
        self.unit = func.get_unit(self.flux)

    def flux_unit(self):
        return self.flux / self.unit

    def err_unit(self):
        return self.err / self.unit

    def mask(maskwindow):
        """
        return a new spectrum object after masking
        """
        arg = func.mask(self.wave)
        new_wave = self.wave[arg]
        new_flux = self.flux[arg]
        new_err = self.err[arg]
        return Spectrum(new_wave, new_flux, new_err, self.filename)


def read_sdss(fname, lw=-float('inf'), rw=float('inf')):
    fit = fits.open(fname)
    data = fit[1].data
    wave = (np.power(10, data['loglam'])).astype(np.float64)
    flux = (data['flux'] * 1.0e-17).astype(np.float64)
    err = (1/data['ivar'] * 1.0e-17).astype(np.float64)
    arg = np.where((wave > lw) & (wave < rw))
    new_wave = wave[arg]
    new_flux = flux[arg]
    new_err = err[arg]
    return new_wave, new_flux, new_err


def read_iraf(fname):
    fit = fits.open(fname)
    data = fit[0].data
    head = fit[0].header
    size = head['NAXIS1']
    begin = head['CRVAL1']
    step = head['CD1_1']
    wave = np.arange(size)*step + begin
    flux = data[0, 0, :].astype(np.float64)
    err = data[3, 0, :].astype(np.float64)
    return wave, flux, err


def read_template(fname):
    fit = fits.open(fname)
    data = fit[1].data
    wave = (10**data['LogLam']).astype(np.float64)
    flux = data['Flux'].astype(np.float64)
    err = data['PropErr'].astype(np.float64)
    return wave, flux, err
