This directory contains alternate files derived from the NANOGrav
nine-year data set used in: 
  Matthews et al 2016, arXiv:1509.08982,
  "The NANOGrav Nine-year Data Set: Astrometric Measurements of 37 Millisecond Pulsars"

Five pulsars have long stretches of single-narrowband-receiver TOAs in the
NANOGrav nine-year data sets.  The TOAs from those times were excised from
the following .tim files.  The parameters and noise-models were then re-fit
using the standard nine-year data analysis procedures, resulting in the
following .par files.

  B1953+29_NANOGrav_9yv1_dual.par
  B1953+29_NANOGrav_9yv1_dual.tim
  J1741+1351_NANOGrav_9yv1_dual.par
  J1741+1351_NANOGrav_9yv1_dual.tim
  J1853+1303_NANOGrav_9yv1_dual.par
  J1853+1303_NANOGrav_9yv1_dual.tim
  J1910+1256_NANOGrav_9yv1_dual.par
  J1910+1256_NANOGrav_9yv1_dual.tim
  J1944+0907_NANOGrav_9yv1_dual.par
  J1944+0907_NANOGrav_9yv1_dual.tim

One pulsar had physically implausible secular changes in the Laplace-Lagrante
parameters (EPS1DOT and EPS2DOT in tempo).  These were removed from the parameter
file.  Analysis of the new file found that Shapiro delay was now signficant,
and XDOT (formerly part of the parameter file) was no longer significant.
The following .par file includes those changes:

  J2317+1439_NANOGrav_9yv1_shapiro.par
