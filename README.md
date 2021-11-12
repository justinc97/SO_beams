# SO_Beam_Profiles
Simple simulation of Gaussian beam convolved CMB maps with calibration sources within the survey regions and added instrument noise.

Noise simulations are ran on the fly and should be of equal number to the CMB realisations.
Recommend 20+ realisations for good results.
ackage_testing jupyter notebook runs through an example of useage.

Really the only necessary input is the tube (PySm versions so 'LT2_MFF2' = 145 GHz), and the number of independent realisations.

Figures folder contains example figures created with outputs of this code.
