import scipy
from time import time
from pathlib import Path
from astropy.cosmology import Planck15
import numpy as np
import matplotlib.pyplot as plt
import pysm3.units as u
import mapsims
from scipy import fftpack
import scipy.stats as stats
import healpy as hp
import pandas as pd
import test
import cv2 

def change_coord(m, coord):
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))
    rot = hp.Rotator(coord=reversed(coord))
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)
    return m[..., new_pix]

def gauss_beam(flatmap, fwhm, flux):
    size = flatmap.shape[0]
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sig = fwhm / (8 * np.log(2)) ** 0.5
    A = (1/(np.sqrt(2*np.pi*sig**2)))
    gaus2d = A * (np.exp(-((x - x0) ** 2 / (2 * sig ** 2) + (y - y0) ** 2 / (2 * sig ** 2))))
    beam = flux * gaus2d
    return beam

def radialprofile(data, r_bins):
    cx = data.shape[1]//2
    cy = data.shape[0]//2
    a, b = data.shape

    [X,Y] = np.meshgrid(np.arange(b) - cx, np.arange(a) - cy)
    R = np.sqrt(X**2 + Y**2)


    profile_mean = []
    profile_std = []
    n_pix = []
    bin_size = 1
    # Use binsize to mask image, average masked values, assign intensity index
    for i in range(len(r_bins)-1):
        mask = (np.greater(R, r_bins[i] - bin_size) & np.less(R, r_bins[i] + bin_size))
        values = data[mask]
        profile_mean.append(np.mean(values))
        profile_std.append(np.std(values))
        n_pix.append(len(values))
    return n_pix, profile_mean, profile_std

def convolve_map_with_gaussian_beam(N, pix_size, beam_size_fwhp, Map):
    "convolves a map with a Gaussian beam pattern.  NOTE: pix_size and beam_size_fwhp need to be in the same units" 
    # make a 2d gaussian 
    #print('Original map sum %0.3f' % (np.sum(Map)))
    gaussian = make_2d_gaussian_beam(N, pix_size, beam_size_fwhp)
  
    # do the convolution
    FT_gaussian = np.fft.fft2(np.fft.fftshift(gaussian)) # first add the shift so that it is central
    FT_Map = np.fft.fft2(np.fft.fftshift(Map)) #shift the map too
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(FT_gaussian*FT_Map))) 
    #print('Convolved map sum %0.3f' % (np.sum(convolved_map)))

    # return the convolved map
    return(convolved_map)

def make_2d_gaussian_beam(N,pix_size,beam_size_fwhp):
     # make a 2d coordinate system
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
    #plt.title('Radial co ordinates')
    #plt.imshow(R)
  
    # make a 2d gaussian 
    beam_sigma = beam_size_fwhp / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.)
    #gaussian = gaussian / np.sum(gaussian)
    gaussian = gaussian / (np.sum(gaussian) * (pix_size ** 2))
    # return the gaussian
    #plt.imshow(gaussian)
    return(gaussian)

def flux2uKcmb(input_flux, pixarea, centre_freq):
    flux = input_flux * u.mJy       # mJy
    flux = flux.to(u.MJy) / pixarea # MJy/sr
    flux = flux.to(u.uK_CMB, 
                   equivalencies = u.cmb_equivalencies(centre_freq * u.GHz))
    return flux

def flux2mjysr(input_flux, pixarea, centre_freq):
    flux = input_flux * u.mJy       # mJy
    flux = flux.to(u.MJy) / pixarea # MJy/sr
    return flux

def uKcmb2mjysr(input_map, centre_freq):
    ukcmb = input_map * u.uK_CMB
    flux = ukcmb.to(u.MJy/u.sr, 
                   equivalencies = u.cmb_equivalencies(centre_freq * u.GHz))
    return flux

def emodes(data):
    covM = data.T.cov()
    e_vals, e_vecs = np.linalg.eigh(covM)
    return e_vals, e_vecs

def Plot_CMB_Map(Map_to_Plot,c_min,c_max,X_width,Y_width):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    print("map mean:",np.mean(Map_to_Plot),"map rms:",np.std(Map_to_Plot))
    plt.gcf().set_size_inches(10, 10)
    im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
    im.set_clim(c_min,c_max)
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    #cbar = plt.colorbar()
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel('angle $[^\circ]$')
    plt.xlabel('angle $[^\circ]$')
    cbar.set_label('tempearture [uK]', rotation=270)
    
    plt.show()
    return(0)