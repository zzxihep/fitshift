#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import specio
import libccf
import template


def only_blue():
    lstname = '/home/zzx/workspace/data/xiamen/P200-Hale_spec/spec2/F0/name.lst'
    dirname = os.path.dirname(lstname)
    lst = open(lstname).readlines()
    bluelst = [i.split()[0] for i in lst]
    redlst = [i.split()[1] for i in lst]
    tempname = 'data/templates/F0_+0.5_Dwarf.fits'
    templat = template.Model(tempname)
    # spec1 = specio.Spectrum(objname1)
    # spec2 = specio.Spectrum(objname2)
    maskwindow = [
        [6270.0, 6320.0],
        [6860.0, 6970.0],
        [7150.0, 7340.0]
    ]
    for i in range(len(bluelst)):
        specb = specio.Spectrum(dirname+os.sep+bluelst[i])
        print(specb.filename)
        # specr = specio.Spectrum(dirname+os.sep+redlst[i])
        # plt.plot(specb.wave, specb.flux)
        # plt.plot(specr.wave, specr.flux)
        # plt.show()
        out = template.fit(templat, specb.wave, specb.flux_unit, specb.err_unit, show=True)


def main():
    lstname = '/home/zzx/workspace/data/xiamen/P200-Hale_spec/spec2/F0/name.lst'
    dirname = os.path.dirname(lstname)
    lst = open(lstname).readlines()
    bluelst = [i.split()[0] for i in lst]
    redlst = [i.split()[1] for i in lst]
    tempname = 'data/templates/F0_+0.5_Dwarf.fits'
    templat = template.Model(tempname)
    # spec1 = specio.Spectrum(objname1)
    # spec2 = specio.Spectrum(objname2)
    maskwindow = [
        [6270.0, 6320.0],
        [6860.0, 6970.0],
        [7150.0, 7340.0]
    ]
    for i in range(len(bluelst)):
        specb = specio.Spectrum(dirname+os.sep+bluelst[i])
        specr = specio.Spectrum(dirname+os.sep+redlst[i])
        # plt.plot(specb.wave, specb.flux)
        # plt.plot(specr.wave, specr.flux)
        # plt.show()
        out = template.fit2(templat, specb, specr, mask=maskwindow)


if __name__ == "__main__":
    only_blue()