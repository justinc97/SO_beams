import healpy as hp
import pysm3.units as u
import matplotlib.pyplot as plt
import cv2 
import utils
import scipy.stats as stats
import scipy
import numpy as np

def beam_cutout(cmb_sims, noise_sims, beam_arcm, flux2use, coordinates, freq, cmb_nside = 4096, shape_out = (512, 512)):    
    """
    This will take full sky cmb and noise simulations with a given beam, frequency and Nside and cut out 512x512 square arrays around coordiantes of given calibration sources
    cmb_sims    - HEALPix format cmb maps in galactic coordinates
    noise_sims  - As above
    beam_arcm   - Beam FWHM in arcminutes
    flux2use    - flux density of calibration source(s) in mJy
    coordinates - Galactic coordinates ot calibration source(s) in format (lon,lat)
    freq        - frequency of tube/beam
    cmb_nside   - Nside of cmb sim (default 4096)
    shape_out   - Shape of cutout (default 512 x 512)
    """

    maps = cmb_sims
    mapglons, mapglats = coordinates
    pixsize = hp.nside2resol(cmb_nside, arcmin = True) * u.arcmin
    pixarea = hp.nside2pixarea(cmb_nside) * u.sr
    num_rels = np.arange(len(maps))

    # Set up empty lists to save various points of map projections if need
    flux_density_maps, flux_density_noise = [], []
    sim_maps, sim_noise, beam_maps = [], [], []

    # Iterate through number of calibration sources given
    for flux in range(len(flux2use)):
        bl_maps, IMAPs, INOISEs, base_maps, base_noises = [], [], [], [] ,[]
        # Iterate through number of map realisations
        for j in num_rels:
            try:
                # Create 512 x 512 cutout around source location
                proj_map = hp.gnomview(map   = cmb_sims[j].data,
                                       xsize = shape_out[0],
                                       ysize = shape_out[0],
                                       reso  = pixsize.value,
                                       rot   = (mapglons, mapglats),
                                       return_projected_map = True,
                                       no_plot = True)
                # Ensure map is fully in survey area
                no_empty = all(i >= hp.UNSEEN for i in proj_map.flatten())
                if no_empty == True:
                    # Create cutout around source location for noise sim
                    """
                    Depending on how you've done noise you might want to include 
                    .data after noise_sims[j] below
                    """
                    proj_noise = hp.gnomview(map = noise_sims[j],
                                             xsize = shape_out[0],
                                             ysize = shape_out[0],
                                             reso = pixsize.value,
                                             rot = (mapglons, mapglats),
                                             return_projected_map = True,
                                             no_plot = True)

                    # Save map versions if need be
                    base_map = proj_map.copy()
                    base_maps.append(base_map)
                    base_noise = proj_noise.copy()
                    base_noises.append(base_noise)

                    # Convert to MJy/sr
                    intensity_map = utils.uKcmb2mjysr(base_map, freq.value)
                    intensity_noise = utils.uKcmb2mjysr(base_noise, freq.value)

                    # Ensure beam to pixel size is appropriate
                    if beam_arcm.value < 5:
                        upres_cmb = cv2.resize(intensity_map, (4096, 4096),
                                               interpolation = cv2.INTER_CUBIC)
                        upres_noise = cv2.resize(intensity_noise, (4096, 4096),
                                                 interpolation = cv2.INTER_CUBIC)

                        upres_pixsize = pixsize / (4096 / 512)
                        upres_pixarea = pixarea / (4096 / 512)

                        # Again save map versions for posterity 
                        IMAPs.append(cv2.resize(intensity_map, (4096, 4096), 
                                           interpolation = cv2.INTER_CUBIC))
                        INOISEs.append(cv2.resize(intensity_noise, (4096, 4096),
                                                 interpolation = cv2.INTER_CUBIC))

                        # Add Source
                        cy = cx = upres_cmb.shape[0] // 2
                        flux_conv = (flux2use[flux] * u.mJy).to(u.MJy) / upres_pixarea
                        upres_cmb[cy, cx] = flux_conv.value

                        # Convolve with Gaussian Beam
                        beam_conv = utils.convolve_map_with_gaussian_beam(4096,
                                                                       upres_pixsize.value, 
                                                                       beam_arcm.value,
                                                                       upres_cmb)

                        # Add Noise to Beam convolved map
                        beam_conv += upres_noise

                    # If large beams
                    elif beam_arcm.value > 5:
                        cy = cx = intensity_map.shape[0] // 2
                        IMAPs.append(intensity_map)
                        INOISESs.append(intensity_noise)

                        # Add Source
                        flux_conv = (flux2use[flux] * u.mJy).to(u.MJy) / pixarea
                        source_map = intensity_map.copy()
                        source_map[cy, cx] = flux_conv.value

                        # Convolve with Gaussian Beam
                        beam_conv = utils.convolve_map_with_gaussian_beam(shape_out[0], 
                                                                          pixsize.value,
                                                                          beam_arcm.value,
                                                                          source_map)

                        # Add Noise to Beam convolved map
                        beam_conv += intensity_noise

                    bl_maps.append(beam_conv)

            # If map includes null regions
            except ValueError:
                pass
            if len(bl_maps) == num_rels:
                break

        # Return what you want but for base all you need really is beam_maps
        sim_maps.append(base_maps)
        sim_noise.append(base_noises)
        beam_maps.append(bl_maps)
        flux_density_maps.append(IMAPs)
        flux_density_noise.append(INOISEs)
        
    return beam_maps

def beam_profile(beam_maps, beam_arcm, flux2use, cmb_nside = 4096, sigmaint = 0.68):
    """
    Calculate beam transfer function by measuring radial profile from source out to
    ~ 10 x beam fwhm and then use healpy beam2bl to perform spherical harmonic
    transform and obtain B_ell
    
    beam_maps - 2D arrays of beam convolved, noise added maps made from previous func
    flux2use  - flux density of calibration source(s)
    cmb_nside - Nside of CMB simulation (default = 4096)
    sigmaint  - interval for confidence levels (default = 0.68 or 1 sigma)
    """
    
    # Set up radial profile
    lmax = 3 * cmb_nside - 1
    pixsize = hp.nside2resol(cmb_nside, arcmin = True) * u.arcmin
    steps = 500
    
    # Again just ensure beam to pixel size is enough
    if beam_arcm.value < 5:
        upres_pixsize = pixsize / (4096 / 512)
        r_max = beam_arcm.value * 10
        res = r_max / steps
        r_bins = np.linspace(0, r_max, steps)
        thet = np.array(((r_bins[:-1] 
                          * upres_pixsize.value + res * upres_pixsize.value/2) 
                         * u.arcmin).to(u.rad) / u.rad) 
        r_prof_bins = r_bins/upres_pixsize.value

    elif beam_arcm.value > 5:
        r_max = beam_arcm.value * 10
        res = r_max / steps
        r_bins = np.linspace(0, r_max, steps)
        thet = np.array(((r_bins[:-1] 
                          * pixsize.value + res * pixsize.value/2) 
                         * u.arcmin).to(u.rad) / u.rad) 
        r_prof_bins = r_bins/pixsize.value
        
    flux_bls, flux_CIs = [], []
    all_bls = []
    radial_profiles = []
    
    # Iterate through calibration sources
    for flux in range(len(flux2use)):
        mean_profs, std_profs = [], []
        bls = []
        for i in range(len(beam_maps[flux])):
            # Calculate radial profile
            n_pix, meanp, stdp = utils.radialprofile(beam_maps[flux][i], r_prof_bins)
            mean_profs.append(meanp/meanp[0])
            std_profs.append(stdp)
            
            # Calculate beam profile
            bl = hp.beam2bl(meanp/meanp[0], (r_bins[:-1] * u.arcmin).to(u.rad).value, lmax)
            
            # Calculate pivot ell from radial profile radius
            pivotell = 180 / (r_max / 60)
            bls.append(bl)
        
        mean_bl = np.mean(bls, axis = 0)
        deg_free = np.array(bls).size - 1
        confidence_level = sigmaint
        std_bl = scipy.stats.sem(bls, axis = 0)
        confidence_int = scipy.stats.t.interval(confidence_level, deg_free,
                                                mean_bl, std_bl)
        radial_profiles.append(mean_profs)
        all_bls.append(bls)
        flux_bls.append(mean_bl)
        flux_CIs.append(confidence_int)
        
    return radial_profiles, all_bls, flux_bls, flux_CIs, pivotell