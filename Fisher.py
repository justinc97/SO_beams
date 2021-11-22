import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import camb
import scipy
from scipy import linalg
import pandas as pd
from matplotlib.patches import Ellipse
from collections import OrderedDict
import healpy as hp
import astropy.units as u
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import fishchips.util
import pickle
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_1 as so_models

cmboutscale = 7.43e12

def get_bin_def(bin_width = 25, lmin = 2, lmax = 5000, 
                bin_width_res = 100, pol = False):
    """
    Create bins for bandpowers.
    """
    nbins_cmb = 0
    npol = (6 if tbeb else 4) if pol else 1
    bin_width = np.asarray(bin_width).ravel()[:npol]
    specs = ['tt', 'ee', 'bb', 'te', 'eb', 'tb'][:npol]
    bin_def = OrderedDict()
    for spec, bw in zip(specs, bin_width):
        bins = np.arange(lmin, lmax, bw)
        bins = np.append(bins, lmax + 1)
        bin_def['cmb_{}'.format(spec)] = np.column_stack((bins[:-1],
                                                         bins[1:]))
        nbins_cmb += len(bins) - 1
    return bin_def

def Cl_Noise(fsky, Bl, t_obs, s2, bands):
    """
    Define noise curves as per Tegmark 97
    """
    ellrange = len(Bl)
    Dl = (ellrange * (ellrange + 1)) / (2 * np.pi)
    wm1 = ((4 * np.pi * s2) / ((t_obs * u.year).to(u.second).value))
    noise = (fsky / (Bl** 2)) * wm1
    return [np.mean((Dl * noise)[bands[j][0]:bands[j][1]]) 
            for j in range(len(bands))]

def bandderivs(paramdict, deriv_param, step_size, lmax = 5000, L = 20):
    """
    Calculate bandpower parameter derivatives.
    paramdict   - Dictionary of parameter values to derive
    deriv_param - List of parameter names
    step_size   - Size of parameter steps for derivative
    lmax        - maximum multipole
    L           - bandpower width
    """
    ells = np.arange(lmax + 1)
    Dl_scale = (ells * (ells + 1)) / (2 * np.pi)
    
    paramdict_left = paramdict.copy()
    paramdict_right = paramdict.copy()

    # Step left and right of the original parameter value
    paramdict_left[deriv_param] -= step_size
    paramdict_right[deriv_param] += step_size


    # Create minus cosmology
    deriv_left = camb.set_params(thetastar = paramdict_left['thetastar'],
                              ombh2 = paramdict_left['ombh2'],
                              omch2 = paramdict_left['omch2'],
                              As    = paramdict_left['As'],
                              tau   = paramdict_left['tau'],
                              ns    = paramdict_left['ns'],
                              lmax  = lmax,
                              WantTransfer = False, 
                              dark_energy_model = 'fluid')
    # Create plus cosmology
    deriv_right = camb.set_params(thetastar = paramdict_right['thetastar'],
                              ombh2 = paramdict_right['ombh2'],
                              omch2 = paramdict_right['omch2'],
                              As    = paramdict_right['As'],
                              tau   = paramdict_right['tau'],
                              ns    = paramdict_right['ns'],
                              lmax  = lmax,
                              WantTransfer = False, 
                              dark_energy_model = 'fluid')
    
    left_results = camb.get_results(deriv_left)
    right_results = camb.get_results(deriv_right)
    
    left_powers = left_results.get_cmb_power_spectra(deriv_left)
    right_powers = right_results.get_cmb_power_spectra(deriv_right)

    
    left_CL = left_powers['unlensed_scalar'][:, 0] * cmboutscale #* Dl_scale 
    right_CL = right_powers['unlensed_scalar'][:, 0] * cmboutscale #* Dl_scale 
    
    bands = get_bin_def(bin_width = L, lmax = lmax)['cmb_tt']
    
    band_means_left, band_means_right = [], []
    for i in range(len(bands)):
        band_means_left.append(np.mean(left_CL[bands[i][0] : bands[i][1]]))
        band_means_right.append(np.mean(right_CL[bands[i][0] : bands[i][1]]))


    dCL_dthet = (np.array(band_means_right) - np.array(band_means_left)) / (2 * step_size)
    
    return dCL_dthet

def DeltCb(band_cents, Cl, Nl, freq, L = 20, fsky = 0.4):
    """
    Calculate power spectra diagonal covariance matrix
    band_cents - Centre points of bands
    Cl         - power spectra
    Nl         - noise spectra
    freq       - frequency of OT
    L          - bandpower width
    fsky       - fraction of sky covered by telescope
    """
    freqs = {'27': 0, '39': 1, '93': 2,
             '145': 3, '225': 4, '280': 5}
    Cov = [(np.sqrt(2 / 
            ((2 * np.array(band_cents)[i] + 1) * L * fsky)) * (
            (cmboutscale * np.array(Cl)[i]) + np.array(Nl)[freqs[freq]][i])) 
            for i in range(len(band_cents))]
    return pd.DataFrame(np.diag(np.array(Cov)**2))

"""
---------------- PLOTTING --------------
"""
from matplotlib.patches import Ellipse

ALPHA1 = 1.52
ALPHA2 = 2.48
ALPHA3 = 3.44
"""float: These three constants are multiplied with 1D sigma to get 2D 68%/95%/99% contours."""

PLOT_MULT = 4.
"""float: Ratio between plot window and sigma."""

def get_ellipse(par1, par2, params, cov, scale1=1, scale2=1):
    """
    Extract ellipse parameters from covariance matrix.
    Parameters
    ----------
        par1 (string): name of parameter 1
        par2 (string): name of parameter 2
        params (list of strings): contains names of parameters to constrain
        cov (numpy array): covariance matrix
    Return
    ------
        tuple, ellipse a, b, angle in degrees, sigma_x, sigma_y, sigma_xy
    """
    # equations 1-4 Coe 2009. returns in degrees
    # first look up indices of parameters
    pind = dict(zip(params, list(range(len(params)))))
    i1 = pind[par1]
    i2 = pind[par2]
    sigma_x2 = cov[i1, i1] * scale1*scale1
    sigma_y2 = cov[i2, i2] * scale2*scale2
    sigma_xy = cov[i1, i2] * scale1*scale2

    if ((sigma_y2/sigma_x2) < 1e-20) or ((sigma_x2/sigma_y2) < 1e-20):
        a2 = max(sigma_x2, sigma_y2) + sigma_xy**2 / max(sigma_x2, sigma_y2)
        b2 = min(sigma_x2, sigma_y2) - sigma_xy**2 / max(sigma_x2, sigma_y2)
    else:
        a2 = (sigma_x2+sigma_y2)/2. + np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                              sigma_xy**2)
        b2 = (sigma_x2+sigma_y2)/2. - np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                              sigma_xy**2)
    angle = np.arctan(2.*sigma_xy/(sigma_x2-sigma_y2)) / 2.
    if (sigma_x2 < sigma_y2):
        a2, b2 = b2, a2

    return np.sqrt(a2), np.sqrt(b2), angle * 180.0 / np.pi, \
        np.sqrt(sigma_x2), np.sqrt(sigma_y2), sigma_xy

def plot_ellipse(ax, par1, par2, parameters, fiducial, cov,
                 resize_lims=True, positive_definite=[], one_sigma_only=False,
                 scale1=1, scale2=1,
                 kwargs1={'ls': '--'},
                 kwargs2={'ls': '-'},
                 default_kwargs={'lw': 1, 'facecolor': 'none',
                                 'edgecolor': 'black'}):
    """
    Plot 1 and 2-sigma ellipses, from Coe 2009.
    Parameters
    ----------
        ax (matpotlib axis): axis upon which the ellipses will be drawn
        par1 (string): parameter 1 name
        par2 (string): parameter 2 name
        parameters (list): list of parameter names
        fiducial (array): fiducial values of parameters
        cov (numpy array): covariance matrix
        color (string): color to plot ellipse with
        resize_lims (boolean): flag for changing the axis limits
        positive_definite (list of string): convenience input,
            parameter names passed in this list will be cut off at 0 in plots.
        scale1 and scale2 are for plotting scale
    Returns
    -------
        list of float : sigma_x, sigma_y, sigma_xy for judging the size of the
            plotting window
    """
    params = parameters
    pind = dict(zip(params, list(range(len(params)))))
    i1 = pind[par1]
    i2 = pind[par2]
    a, b, theta, sigma_x, sigma_y, sigma_xy = get_ellipse(
        par1, par2, params, cov, scale1, scale2)

    fid1 = fiducial[i1] * scale1
    fid2 = fiducial[i2] * scale2

    # use defaults and then override with other kwargs
    kwargs1_temp = default_kwargs.copy()
    kwargs1_temp.update(kwargs1)
    kwargs1 = kwargs1_temp
    kwargs2_temp = default_kwargs.copy()
    kwargs2_temp.update(kwargs2)
    kwargs2 = kwargs2_temp

    if not one_sigma_only:
        # 2-sigma ellipse
        e1 = Ellipse(
            xy=(fid1, fid2),
            width=a * 2 * ALPHA2, height=b * 2 * ALPHA2,
            angle=theta, **kwargs2)
        ax.add_artist(e1)
        e1.set_clip_box(ax.bbox)

    # 1-sigma ellipse
    e2 = Ellipse(
        xy=(fid1, fid2),
        width=a * 2 * ALPHA1, height=b * 2 * ALPHA1,
        angle=theta, **kwargs1)
    ax.add_artist(e2)
    e2.set_alpha(1.0)
    e2.set_clip_box(ax.bbox)

    if resize_lims:
        if par1 in positive_definite:
            ax.set_xlim(max(0.0, -PLOT_MULT*sigma_x),
                        fid1+PLOT_MULT*sigma_x)
        else:
            ax.set_xlim(fid1 - PLOT_MULT * sigma_x,
                        fid1 + PLOT_MULT * sigma_x)
        if par2 in positive_definite:
            ax.set_ylim(max(0.0, fid2 - PLOT_MULT * sigma_y),
                        fid2 + PLOT_MULT * sigma_y)
        else:
            ax.set_ylim(fid2 - PLOT_MULT * sigma_y,
                        fid2 + PLOT_MULT * sigma_y)

    return sigma_x, sigma_y, sigma_xy

def plot_triangle(obs_parameters, obs_fiducial, cov, f=None, ax=None, color='black', positive_definite=[],
                  labels=None, scales=None):
    """
    Makes a standard triangle plot using the fishchips `obs` and `cov` objects.
    Args:
        f (optional,matplotlib figure): pass this if you already have a figure
        ax (optional,matplotlib axis): existing axis
    """
    return plot_triangle_base(
        obs_parameters, obs_fiducial, cov, f=f, ax=ax,
        positive_definite=positive_definite,
        labels=labels, scales=scales,
        ellipse_kwargs1={'ls': '--', 'edgecolor': color},
        ellipse_kwargs2={'ls': '-', 'edgecolor': color},
        color_1d=color)

def plot_triangle_base(params, fiducial, cov, f=None, ax=None,
                       positive_definite=[],
                       labels=None, scales=None,
                       ellipse_kwargs1={'ls': '--', 'edgecolor': 'black'},
                       ellipse_kwargs2={'ls': '-', 'edgecolor': 'black'},
                       xlabel_kwargs={'labelpad': 30},
                       ylabel_kwargs={},
                       fig_kwargs={'figsize': (12, 12)},
                       color_1d='black'):
    """
    Makes a standard triangle plot.
    Parameters
    ----------
        params (list):
            List of parameter strings
        fiducial (array):
            Numpy array consisting of where the centers of ellipses should be
        cov : numpy array
            Covariance matrix to plot
        f : optional, matplotlib figure
            Pass this if you already have a figure
        ax : array containing matplotlib axes
            Pass this if you already have a set of matplotlib axes
        labels : list
            List of labels corresponding to each dimension of the covariance matrix
        scales : list
            It's sometimes nice to scale a parameter by 10^9 or something. Pass this
            array, where each index corresponds to each parameter, to do this. Nice for
            i.e. plotting A_s. If you don't pass anything, it won't scale anything.
        ellipse_kwargs1 : dict
            Keyword arguments for passing to the 1-sigma Matplotlib Ellipse call. You 
            can change this to change the color of your ellipses, for example. 
        ellipse_kwargs2 : dict
            Keyword arguments for passing to the 2-sigma Matplotlib Ellipse call. You 
            can change this to change the color of your ellipses, for example. 
        xlabel_kwargs : dict
            Keyword arguments which are passed to `ax.set_xlabel()`. You can change the
            color and font-size of the x-labels, for example. By default, it includes
            a little bit of label padding.
        ylabel_kwargs : dict
            Keyword arguments which are passed to `ax.set_xlabel()`. You can change the
            color and font-size of the x-labels, for example. By default, it includes
            a little bit of label padding.
    Returns
    -------
        fig, ax
            matplotlib figure and axis array
    """

    nparams = len(params)
    if scales is None:
        scales = np.ones(nparams)

    if ax is None or f is None:
        print('generating new axis')
        f, ax = plt.subplots(nparams, nparams, **fig_kwargs)

    if labels is None:
        labels = [(r'$\mathrm{' + p.replace('_', r'\_') + r'}$')
                  for p in params]
    print(labels)
    # stitch together axes to row=nparams-1 and col=0
    # and turn off non-edge
    for ii in range(nparams):
        for jj in range(nparams):
            if ii == jj:
                ax[jj, ii].get_yaxis().set_visible(False)
                if ii < nparams-1:
                    ax[jj, ii].get_xaxis().set_ticks([])

            if ax[jj, ii] is not None:
                if ii < jj:
                    if jj < nparams-1:
                        ax[jj, ii].set_xticklabels([])
                    if ii > 0:
                        ax[jj, ii].set_yticklabels([])

                    if jj > 0:
                        # stitch axis to the one above it
                        if ax[0, ii] is not None:
                            ax[jj, ii].get_shared_x_axes().join(
                                ax[jj, ii], ax[0, ii])
                    elif ii < nparams-1:
                        if ax[jj, nparams-1] is not None:
                            ax[jj, ii].get_shared_y_axes().join(
                                ax[jj, ii], ax[jj, nparams-1])

    # call plot_ellipse
    for ii in range(nparams):
        for jj in range(nparams):
            if ax[jj, ii] is not None:
                if ii < jj:
                    plot_ellipse(ax[jj, ii], params[ii],
                                 params[jj], params, fiducial, cov,
                                 positive_definite=positive_definite,
                                 scale1=scales[ii], scale2=scales[jj],
                                 kwargs1=ellipse_kwargs1,
                                 kwargs2=ellipse_kwargs2)
                    if jj == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], **xlabel_kwargs)
                        for tick in ax[jj, ii].get_xticklabels():
                            tick.set_rotation(45)
                    if ii == 0:
                        ax[jj, ii].set_ylabel(labels[jj], **ylabel_kwargs)
                elif ii == jj:
                    # plot a gaussian if we're on the diagonal
                    sig = np.sqrt(cov[ii, ii])
                    if params[ii] in positive_definite:
                        grid = np.linspace(
                            fiducial[ii],
                            fiducial[ii] + PLOT_MULT * sig, 100)
                    else:
                        grid = np.linspace(
                            fiducial[ii] - PLOT_MULT*sig,
                            fiducial[ii] + PLOT_MULT*sig, 100)
                    posmult = 1.0
                    if params[ii] in positive_definite:
                        posmult = 2.0
                    ax[jj, ii].plot(grid * scales[ii],
                                    posmult * np.exp(
                                        -(grid-fiducial[ii])**2 /
                                        (2 * sig**2)) / (sig * np.sqrt(2*np.pi)),
                                    '-', color=color_1d)
                    ax[jj, ii].set_ylim(bottom=0)
                    if ii == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], **xlabel_kwargs)
                        for tick in ax[jj, ii].get_xticklabels():
                            tick.set_rotation(45)
                else:
                    ax[jj, ii].axis('off')

    return f, ax