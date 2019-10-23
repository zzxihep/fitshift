import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from lmfit import report_fit
import click


import template
import specio
import func


template_path = os.path.join('data', 'templates')


def get_template_lst():
    namelst = glob.glob(template_path+os.sep+'*.fits')
    namelst = sorted(namelst)
    templatelst = [template.Model(name) for name in namelst]
    return templatelst


@click.command()
@click.option('--name', prompt='The spectrum file name', 
              help='The spectrum file name with path')
def find_type(name):
    ''' a ugly script used to find the stellar type'''
    modellst = get_template_lst()
    objname = name
    wave, flux, err = specio.read_iraf(objname)
    unit = func.get_unit(flux)
    flux = flux / unit
    err = err / unit
    resultlst, chisqlst = [], []
    result = template.fit(modellst[0], wave, flux, err)
    for model in modellst:
        # print(model.filename)
        result = template.fit(model, wave, flux, err, params=result.params)
        text = '%.40s  %.4f  %.4f' % (model.filename, result.chisqr, result.redchi)
        print(text)
        resultlst.append(result)
        chisqlst.append(result.redchi)
    
    arg = np.argmin(chisqlst)
    bestresult = resultlst[arg]
    bestmodel = modellst[arg]
    text = '%s  %f' % (bestmodel.filename, bestresult.redchi)
    print(text)
    report_fit(bestresult)
    parscale, parsigma, parshift = bestmodel.get_pars(bestresult.params)
    spec_fit = bestmodel.get_spectrum(bestresult.params, wave)
    scale = bestmodel.get_scale(wave, parscale)
    spec_fit2 = spec_fit / scale
    spec_data = flux / scale
    plt.plot(wave, spec_data)
    plt.plot(wave, spec_fit2)
    plt.figure()
    plt.plot(wave, flux)
    plt.plot(wave, spec_fit)
    plt.figure()
    plt.plot(wave, scale)
    plt.show()
    # bestname = 'data/templates/F0_+0.5_Dwarf.fits'


if __name__ == "__main__":
    find_type()