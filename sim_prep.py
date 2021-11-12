import mapsims
import numpy as np
import utils
from pathlib import Path
import pysm3.units as u
import healpy as hp

"""
If using PySm3 CMB and noise sims
"""

channel_dict = {
    'LT0_UHF1' : 0, 
    'LT0_UHF2' : 1,
    'LT1_UHF1' : 2,
    'LT1_UHF2' : 3,
    'LT2_MFF1' : 4,
    'LT2_MFF2' : 5,
    'LT3_MFF1' : 6,
    'LT3_MFF2' : 7,
    'LT4_MFS1' : 8,
    'LT4_MFS2' : 9,
    'LT5_MFS1' : 10,
    'LT5_MFS2' : 11,
    'LT6_LF1'  : 12,
    'LT6_LF2'  : 13,
    'ST0_UHF1' : 14,
    'ST0_UHF2' : 15,
    'ST1_MFF1' : 16,
    'ST1_MFF2' : 17,
    'ST2_MFS1' : 18,
    'ST2_MFS2' : 19,
    'ST3_LF1'  : 20,
    'ST3_LF2'  : 21}

def read_cmb_maps(OT, num_rels, resol = 4096, coord = 'gal'):
    """
    Read some number of CMB simulations
    """
    
    channels = mapsims.parse_channels()
    channel = channels[channel_dict[str(OT)]]
    print("{} has a beam of {} ({}) and a center frequency at {}".format(channel, 
              channel.beam, channel.beam.to(u.deg), channel.center_frequency))
    cmb_nside = resol
    beam_arcm = channel.beam 
    
    cmb_SO_region = mapsims.SONoiseSimulator(nside = cmb_nside)
    hitmaps, ave_nhits = cmb_SO_region.get_hitmaps(tube = channel.tube)
    
    # Read CMB realisations
    cmb_folder = '/project/projectdirs/sobs/v4_sims/mbs/201911_lensed_cmb'
    cmb_filename_template = "{nside}/{content}/{num:04d}/simonsobs_{content}_uKCMB_{telescope}{band}_nside{nside}_{num:04d}.fits"
    cmb_filenames = [Path(cmb_folder) / cmb_filename_template.format(nside = cmb_nside, 
                     content = "cmb", num = num, telescope = channel.telescope.lower(), 
                     band = channel.band) for num in range(num_rels)]

    cmb_maps = [hp.ma(hp.read_map(filename, 0, dtype = np.float64)) for filename in cmb_filenames] # Full sky CMB map (need to cut to SO observing region using hitmaps)
  
    cmb_SO_region = mapsims.SONoiseSimulator(nside = cmb_nside)
    hitmaps, ave_nhits = cmb_SO_region.get_hitmaps(tube = channel.tube)

    for each in cmb_maps:
        each[hitmaps[0]==0]=hp.UNSEEN

    if coord == 'gal':
        rels = []
        for each in cmb_maps:
            rels.append(utils.change_coord(each, ['E', 'G']))
    else:
        rels = cmb_maps
    
    constants = (beam_arcm, cmb_nside, channel.center_frequency, channel)
    return rels, constants

def simnoise(OT, resol = 4096, coord = 'gal'):
    """
    Run noise sims on the fly with given OT (Optical Tube).
    By default we want resolution of Nside = 4096 and Galactic Coordinates
    """
    channels = mapsims.parse_channels()
    channel = channels[channel_dict[str(OT)]]
    sim = mapsims.from_config(["common.toml", "noise.toml"],
                          override={"channels":"tube:"+channel.tube, "output_folder":".", "num":1})
    noise = sim.other_components["noise"]
    maps = sim.execute()
    maps_TT = maps[OT][0]
    maps_dg = hp.ud_grade(maps_TT, resol)
    if coord == 'gal':
        noise_map = utils.change_coord(maps_dg, ['E', 'G'])
    else:
        noise_map = maps_dg
    return noise_map

