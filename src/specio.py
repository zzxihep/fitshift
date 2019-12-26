import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import func


def spec_creator(fname):
    """
    get the origin of the spectrum, the origin include
    ['text', 'iraf', 'lamost', 'sdss', 'template', 'unknow']

    """
    extension = os.path.splitext(fname)[1]
    if extension != '.fits' and extension != '.fit':
        return 'text'
    head = fits.getheader(fname)
    if 'ORIGIN' in head:
        value = head['ORIGIN']
        if 'NOAO-IRAF' in value:
            return 'iraf'
        if value == 'NAOC-LAMOST':
            return 'lamost'
    if 'TELESCOP' in head:
        if 'SDSS' in head['TELESCOP']:
            return 'sdss'
    fit = fits.open(fname)
    if len(fit) > 1 and 'PropErr' in fit[1].header.values():
        return 'template'
    if 'CREATOR' in head:
        return head['CREATOR'].replace(' ', '')
    fit.close()
    return 'unknow'


class Spectrum:
    def __init__(self, filename, hduid=None):
        self.filename = filename
        self.hduid = hduid
        origin = spec_creator(filename)
        if origin == 'lamost':
            self.read_lamost(filename, hduid)
        elif origin == 'iraf':
            self.read_iraf(filename)
        elif origin == 'text':
            self.read_text(filename)
        elif origin == 'sdss':
            self.read_sdss(filename)
        elif origin == 'template':
            self.read_template(filename)
        elif origin == 'Pleinpot2':
            self.read_pleinpot2(filename)
        self.set_unit()

    def clean_cosmic_ray(self):
        self.flux = func.median_reject_cos(self.flux)
        self.set_unit()

    def read_pleinpot2(self, fname):
        fit = fits.open('lte047.fits')
        tdata = fit[1].data
        self.wave = tdata['WAVELENGTH'].astype(np.float64)
        self.flux = tdata['FLUX'].astype(np.float64)
        self.err = None
        self.header = fit[1].header
        self.data = fit[1].data

    def read_lamost(self, filename, hduid):
        fit = fits.open(filename)
        _hduid = hduid
        if hduid is None:
            _hduid = 1
        hdu = fit[_hduid]
        data = hdu.data
        self.wave = 10**data['LOGLAM'].astype(np.float64)
        self.flux = data['FLUX'].astype(np.float64)
        self.err = 1/data['IVAR'].astype(np.float64)
        arg = np.argsort(self.wave)
        self.wave = self.wave[arg]
        self.flux = self.flux[arg]
        self.err = self.err[arg]
        self.header = hdu.header
        self.data = hdu.data

    def read_iraf(self, fname):
        fit = fits.open(fname)
        data = fit[0].data
        head = fit[0].header
        size = head['NAXIS1']
        begin = head['CRVAL1']
        step = head['CD1_1']
        self.wave = np.arange(size)*step + begin
        self.flux = data[0, 0, :].astype(np.float64)
        self.err = data[3, 0, :].astype(np.float64)
        self.header = head
        self.data = data

    def read_sdss(self, fname):
        fit = fits.open(fname)
        data = fit[1].data
        head = fit[1].header
        self.wave = (np.power(10, data['loglam'])).astype(np.float64)
        self.flux = (data['flux'] * 1.0e-17).astype(np.float64)
        self.err = (1/data['ivar'] * 1.0e-17).astype(np.float64)
        self.header = head
        self.data = data

    def read_template(self, fname):
        fit = fits.open(fname)
        data = fit[1].data
        head = fit[1].header
        self.wave = (10**data['LogLam']).astype(np.float64)
        self.flux = data['Flux'].astype(np.float64)
        self.err = data['PropErr'].astype(np.float64)
        self.header = head
        self.data = data

    def read_text(self, fname):
        data = np.loadtxt(fname)
        self.wave = data[:, 0]
        self.flux = data[:, 1]
        if data.shape[1] == 3:
            self.err = data[:, 2]
        else:
            self.err = data[:, 2]
        self.header = None
        self.data = None

    def set_unit(self):
        self.unit = func.get_unit(self.flux)
        self.flux_unit = self.flux / self.unit
        if self.err is not None:
            self.err_unit = self.err / self.unit
        else:
            self.err_unit = None

    def mask(self, maskwindow):
        """
        return a new spectrum object after masking
        """
        arg = func.mask(self.wave, maskwindow)
        new_wave = self.wave[arg]
        new_flux = self.flux[arg]
        new_err = self.err[arg]
        return Spectrum(new_wave, new_flux, new_err, self.filename)

    def select(self, selectwindow):
        """
        return a new spectrum object the wavelength included in the maskwindow.
        """
        win = selectwindow[0]
        selected = ((self.wave > win[0]) & (self.wave < win[1]))
        for win in selectwindow[1:]:
            selected = (selected | ((self.wave > win[0]) &
                                    (self.wave < win[1])))
        arg = np.where(selected)
        new_wave = self.wave[arg]
        new_flux = self.flux[arg]
        if self.err is not None:
            new_err = self.err[arg]
        else:
            new_err = None
        self.wave = new_wave
        self.flux = new_flux
        self.err = new_err
        self.set_unit()
        newspec = Spectrum(self.filename)
        newspec.wave = new_wave
        newspec.flux = new_flux
        newspec.err = new_err
        newspec.set_unit()
        return newspec

    def show(self):
        plt.plot(self.wave, self.flux)
        plt.show()


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
