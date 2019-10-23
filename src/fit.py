#!/usr/bin/env python

import math
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from lmfit.printfuncs import report_fit
import corner
import rebin
import convol
import specio


def get_unit(flux):
    return 10**math.floor(math.log10(np.median(flux)))


class Model:
    def __init__(self, tmpname):
        wave, flux, err = specio.read_template(tmpname)
        self.wave = wave
        self.flux = flux
        self.err = err
        self.wshift = -(wave[0]+wave[-1])/2
        self.wscale = 1.98/(wave[-1]-wave[0])
        typicalflux = np.median(self.flux)
        exponent = math.floor(math.log10(typicalflux))
        self.unit = 10**exponent
        self.flux = self.flux / self.unit

    def reset_wave_zoom(self, wave):
        self.wshift = -(wave[0]+wave[-1])/2
        self.wscale = 1.99/(wave[-1]-wave[0])

    def trans_wave(self, wave):
        return (wave + self.wshift) * self.wscale

    def get_scale(self, wave, par):
        tmpwave = self.trans_wave(wave)
        return np.array(convol.poly(tmpwave, par))

    def get_legendre_scale(self, wave, par):
        tmpwave = self.trans_wave(wave)
        return np.array(convol.legendre_poly(tmpwave, par))

    def convol_spectrum(self, wave, par):
        # tmpwave = self.trans_wave(wave)
        return np.array(convol.gauss_filter(wave, self.flux, par))

    def get_wave(self, parwave):
        return np.array(convol.map_wave(self.wave, parwave))

    def get_spectrum_pre(self, wave, arrshift, arrsigma, arrscale):
        new_wave = self.get_wave(arrshift)
        new_flux = self.convol_spectrum(new_wave, arrsigma)
        # scale = self.get_scale(new_wave, arrscale)
        scale = self.get_legendre_scale(wave, arrscale)
        flux_rebin = np.array(rebin.rebin(new_wave, new_flux, wave))
        flux_aftscale = flux_rebin * scale
        return flux_aftscale

    def get_spectrum(self, pars, wave):
        parscale, parsigma, parshift = get_pars(pars)
        return self.get_spectrum_pre(wave, parshift, parsigma, parscale)

    def residual(self, pars, x, data, eps=None):
        spec = self.get_spectrum(pars, x)
        if eps is not None:
            return (data - spec) / eps
        else:
            return data-spec


def get_pars(pars):
    """
    return parscale, parsigma, parshift
    """
    pardata = {'scale':[], 'sigma':[], 'shift':[]}
    for key in pars:
        flag = key[:5]
        order = int(key[5:])
        value = pars[key].value
        pardata[flag].append([order, value])
    dicscale = dict(pardata['scale'])
    dicsigma = dict(pardata['sigma'])
    dicshift = dict(pardata['shift'])
    scalemax = max(dicscale)+1
    sigmamax = max(dicsigma)+1
    shiftmax = max(dicshift)+1
    parscale = np.zeros(scalemax, dtype=np.float64)
    parsigma = np.zeros(sigmamax, dtype=np.float64)
    parshift = np.zeros(shiftmax, dtype=np.float64)
    for key in dicscale:
        parscale[key] = dicscale[key]
    for key in dicsigma:
        parsigma[key] = dicsigma[key]
    for key in dicshift:
        parshift[key] = dicshift[key]
    return parscale, parsigma, parshift


def set_pars(pars, prefix, order, valuelst=None, minlst=None, maxlst=None):
    if isinstance(order, int):
        numlst = range(order+1)
    else:
        numlst = order
    if valuelst is None:
        valuelst = [None] * len(numlst)
    if minlst is None:
        minlst = [-float('inf')] * len(numlst)
    if maxlst is None:
        maxlst = [float('inf')] * len(numlst)
    for ind, num in enumerate(numlst):
        keyword = prefix+str(num)
        pars.add(keyword, value=valuelst[ind], min=minlst[ind], max=maxlst[ind])


def show_err(ax, wave, spec, err):
    low = spec-err
    upp = spec+err
    ax.fill_between(wave, low, upp, alpha=0.3, color='grey')

from astropy.constants import c

def main():
    tmpname = 'data/F5_-1.0_Dwarf.fits'
    model = Model(tmpname)
    # residual, get_spec = get_residual(model)
    residual = model.residual
    params = Parameters()
    # set_pars(params, 'shift', [0, 1], valuelst=[1.1568794774016442, -0.0007594668175056121])
    set_pars(params, 'shift', [1], valuelst=[-2.5688e-04])
    set_pars(params, 'sigma', [1], valuelst=[0.00016558594418925043], minlst=[1.0e-8])
    scalevalst = [4.543402040007523, -0.20454792267985503, -0.2391637452260473,
                  0.2190777818642178, -0.09965310075298969, -0.1255319879292037]
    set_pars(params, 'scale', 5, valuelst=scalevalst)

    ftargetname = 'data/spec-4961-55719-0378.fits'
    targetname = '/home/zzx/workspace/data/xiamen/P200-Hale_spec/blue/reduce_second/specdir/fawftbblue0070.fits'
    # new_wo, new_fo, new_eo = specio.read_sdss(ftargetname, lw=3660, rw=10170)
    new_wo, new_fo, new_eo = specio.read_iraf(targetname)
    model.reset_wave_zoom(new_wo)
    unit = get_unit(new_fo)
    new_fo = new_fo / unit
    new_eo = new_eo / unit
    print(unit)

    out = minimize(residual, params, args=(new_wo, new_fo, new_eo),
                   method='leastsq')
    out_parms = out.params
    report_fit(out)
    spec_fit = model.get_spectrum(out_parms, new_wo)
    # plt.errorbar(new_wo, new_fo, yerr=new_eo)
    plt.plot(new_wo, new_fo)
    plt.plot(new_wo, spec_fit)
    show_err(plt.gca(), new_wo, new_fo, new_eo)
    scalepar, parsigma, parshift = get_pars(out_parms)
    print(parshift[1])
    velocity = str(parshift[1]*c.to('km/s'))
    print('shift = '+velocity)
    myscale = model.get_scale(new_wo, scalepar)
    plt.figure()
    plt.plot(new_wo, myscale)
    plt.show()

    emcee_params = out.params.copy()
    # emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(1.0e-20),
    #                  max=np.log(1.0e20))
    result_emcee = minimize(residual, params=emcee_params,
                            args=(new_wo, new_fo, new_eo), method='emcee',
                            nan_policy='omit', steps=1000)# , workers='mpi4py')
    report_fit(result_emcee)
    show_err(plt.gca(), new_wo, new_fo, new_eo)
    plt.plot(new_wo, new_fo)
    # plt.plot(new_wo, spec_fit)
    spec_emcee = model.get_spectrum(result_emcee.params, new_wo)
    plt.plot(new_wo, spec_emcee)
    corner.corner(result_emcee.flatchain, labels=result_emcee.var_names,
                  truths=list(result_emcee.params.valuesdict().values()))
    plt.show()


if __name__ == "__main__":
    main()
