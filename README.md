# Transiting Planets around Hot Stars with TESS
The TOI catalog includes a large number of planet candidates around hot stars (Teff ~> 7000K), which were rare in Kepler and are frequently dropped from spectroscopic follow-up observations due to rapid rotation. The goal for this project is to filter out the most promising TESS hot star planet candidates by calculating revised stellar properties using Gaia, comparing mean stellar densities with the measured transit duration from TESS, refit TESS transits using revised stellar parameters as priors, and brainstorm ideas for how to confirm/validate the best candidates (if there are any left!).

Subprojects (at least 3):
1) Make a list of >7000K TOIs with reasonable quality cuts (transits SNR)
2) Use ExoFOP to exclude known spectroscopic binaries, background EBs, etc
3) Recalculate stellar parameters for host stars (possibly including spectroscopic information from  and compare to transit durations
4) Refit transits for candidates passing steps 1-3, check sample for pulsations
