#!/usr/bin/env python

# import os
import time
import math
import numpy as np
from astropy.io import fits
from astropy.constants import c
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from lmfit.printfuncs import report_fit
import corner
import rebin
import convol
import specio
import func



class Model:
    def __init__(self, tmpname):
        self.filename = tmpname
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

    def set_lmpar_name(self, parscale, parsigma, parshift):
        self.lmscale_name = [None] * (max(parscale)+1)
        for key in parscale:
            self.lmscale_name[key] = parscale[key]
        self.lmsigma_name = [None] * (max(parsigma)+1)
        for key in parsigma:
            self.lmsigma_name[key] = parsigma[key]
        self.lmshift_name = [None] * (max(parshift)+1)
        for key in parshift:
            self.lmshift_name[key] = parshift[key]

    def get_scale_par(self, pars):
        parscale = []
        for parname in self.lmscale_name:
            if parname is None:
                parscale.append(0.0)
            else:
                parscale.append(pars[parname].value)
        return parscale

    def get_sigma_par(self, pars):
        parsigma = []
        for parname in self.lmsigma_name:
            if parname is None:
                parsigma.append(0.0)
            else:
                parsigma.append(pars[parname].value)
        return parsigma

    def get_shift_par(self, pars):
        parshift = []
        for parname in self.lmshift_name:
            if parname is None:
                parshift.append(0.0)
            else:
                parshift.append(pars[parname].value)
        return parshift

    def get_spectrum(self, pars, wave):
        try:
            parscale = self.get_scale_par(pars)
            parsigma = self.get_sigma_par(pars)
            parshift = self.get_shift_par(pars)
        except AttributeError as err:
            print(err)
            print('May be you forget to execute set_lmpar_name function')
            parscale, parsigma, parshift = get_pars_name(pars)
            self.set_lmpar_name(parscale, parsigma, parshift)
            return self.get_spectrum(pars, wave)
        return self.get_spectrum_pre(wave, parshift, parsigma, parscale)

    def residual(self, pars, x, data, eps=None):
        spec = self.get_spectrum(pars, x)
        if eps is not None:
            return (data - spec) / eps
        else:
            return data-spec


def get_pars_name(pars):
    """
    return dictionary of parscale, parsigma, parshift
    """
    pardata = {'scale':[], 'sigma':[], 'shift':[]}
    for key in pars:
        flag = key[:5]
        order = int(key[5:])
        pardata[flag].append([order, key])
    dicscale = dict(pardata['scale'])
    dicsigma = dict(pardata['sigma'])
    dicshift = dict(pardata['shift'])
    return dicscale, dicsigma, dicshift


def set_pars(pars, prefix, order, valuelst=None, minlst=None, maxlst=None):
    """
    add par to pars, the parname is named as prefix+str(order)
    order: list(like [0, 1, 2, 3]) or int (the orderlst will be regarded as
    [0, 1, 2,..., order]),
    return a dictionary(keyword:order, value:parname)
    """
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
    keywordlst = []
    for ind, num in enumerate(numlst):
        keyword = prefix+str(num)
        pars.add(keyword, value=valuelst[ind], min=minlst[ind], max=maxlst[ind])
        keywordlst.append([num, keyword])
    return dict(keywordlst)


def fit(template, wave, flux, err, params=None, show=False, isprint=False):
    if params is None:
        params = Parameters()
        # set_pars(params, 'shift', [0, 1], valuelst=[1.16, -0.00076])
        shiftparname = set_pars(params, 'shift', [1], valuelst=[0.0])
        sigmaparname = set_pars(params, 'sigma', [1], valuelst=[0.001],
                                minlst=[1.0e-8])
        scalevalst = [4.5, -0.2, -0.24, 0.22, -0.1, -0.13]
        scaleparname = set_pars(params, 'scale', 5, valuelst=scalevalst)
        template.set_lmpar_name(scaleparname, sigmaparname, shiftparname)
    template.reset_wave_zoom(wave)
    # start = time.process_time()
    # print(start)
    out = minimize(template.residual, params, args=(wave, flux, err),
                   method='leastsq')
    # end = time.process_time()
    # print("Time used:", end-start)
    if isprint:
        report_fit(out)
    if show:
        spec_fit = template.get_spectrum(out.params, wave)
        lower = flux - err
        upper = flux + err
        plt.figure()
        plt.fill_between(wave, lower, upper, alpha=0.3, color='grey')
        plt.plot(wave, flux)
        plt.plot(wave, spec_fit)
        plt.show()
    return out


def fit2(template, spec1, spec2, mask=None, params=None, show=False, print=False):
    """
    Some spectrum are observed by multichannel instrument. So there are multiple
    spectral files corresponding to one observation. This function try to fit
    the two part of the spectrum simultaneously using 2 scale curves, 1 shift,
    2 sigma pars. the mask window format should be like this
    [[l1, r1], [l2, r2], ...]
    """
    fname = template.filename
    temp2 = Model(fname)


def main():
    tmpname = 'data/F5_-1.0_Dwarf.fits'
    model = Model(tmpname)

    ftargetname = 'data/spec-4961-55719-0378.fits'
    targetname = '/home/zzx/workspace/data/xiamen/P200-Hale_spec/blue/reduce_second/specdir/fawftbblue0070.fits'
    new_wo, new_fo, new_eo = specio.read_sdss(ftargetname, lw=3660, rw=10170)
    unit = func.get_unit(new_fo)
    new_fo = new_fo / unit
    new_eo = new_eo / unit
    # new_wo, new_fo, new_eo = specio.read_iraf(targetname)
    result = fit(model, new_wo, new_fo, new_eo, show=True)

    residual = model.residual

    out_parms = result.params
    spec_fit = model.get_spectrum(out_parms, new_wo)
    scalepar = model.get_scale_par(out_parms)
    parshift = model.get_shift_par(out_parms)
    print(parshift[1])
    velocity = str(parshift[1]*c.to('km/s'))
    print('shift = '+velocity)
    myscale = model.get_scale(new_wo, scalepar)

    plt.figure()
    plt.plot(new_wo, myscale)
    plt.show()

    emcee_params = result.params.copy()
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
