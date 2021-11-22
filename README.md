# SO_Beam_Profiles
Simple simulation of Gaussian beam convolved CMB maps with calibration sources within the survey regions and added instrument noise.

Noise simulations are ran on the fly and should be of equal number to the CMB realisations.
Recommend 20+ realisations for good results.
package_testing jupyter notebook runs through an example of useage.
Fisher Workthrough notebook works through some Fisher calculations and plotting.

Really the only necessary input is the tube (PySm versions so 'LT2_MFF2' = 145 GHz), and the number of independent realisations.

Figures folder contains example figures created with outputs of this code.

## Contains
* `sim_prep`: Read CMB simulations and run noise mapsims on the fly
* `Beam_Profiles`: Mask CMB maps to square region to add source, convolve with Gaussian beam and add noise
* `utils`: Utilities file with extra functions used throughout along with a nice map cutout plot method
* `Fisher`: Containts codes for Fisher calculations and plotting

##
All Code written by Justin Clancy please use appropriately and with permission :) 
