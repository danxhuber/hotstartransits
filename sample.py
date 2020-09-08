import numpy as np
import os, sys
from astropy.io import ascii
import matplotlib.pyplot as plt
import pdb

plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.size']=16
plt.rcParams['mathtext.default']='regular'
plt.rcParams['lines.markersize']=6
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='8'
plt.rcParams['axes.formatter.useoffset']=False

# define the sample

# sigma_depth/depth cut
snrcut=0.1

tess=ascii.read('csv-file-toi-catalog.csv')

tess_spoc=tess[np.where((tess['TOI Disposition'] == 'PC') & \
(tess['Source Pipeline'] == 'spoc') & \
(tess['Transit Depth Error']/tess['Transit Depth Value'] < snrcut))]

tess_qlp=tess[np.where((tess['TOI Disposition'] == 'PC') & \
(tess['Source Pipeline'] == 'qlp') & \
(tess['Transit Depth Error']/tess['Transit Depth Value'] < snrcut))]

kepler=ascii.read('cumulative_2020.09.05_12.25.06.csv')
kepler=kepler[np.where((kepler['koi_pdisposition'] == 'CANDIDATE') & (kepler['koi_depth_err1']/kepler['koi_depth'] < 0.1))]


bs=np.arange(3000,10000,100)
alf=0.4

plt.ion()
plt.clf()

plt.subplot(2,1,2)
plt.hist(tess_qlp['Effective Temperature Value'],bins=bs,color='blue',alpha=alf,density=True)
plt.hist(tess_qlp['Effective Temperature Value'],bins=bs,color='blue',alpha=alf,histtype='step',ls='dashed',lw=3,density=True)
plt.hist(tess_spoc['Effective Temperature Value'],bins=bs,color='black',alpha=alf,density=True)
plt.hist(tess_spoc['Effective Temperature Value'],bins=bs,color='black',alpha=alf,histtype='step',ls='dashed',lw=3,density=True)
plt.hist(kepler['koi_steff'],bins=bs,color='red',alpha=alf,density=True)
plt.hist(kepler['koi_steff'],bins=bs,color='red',alpha=alf,histtype='step',ls='dashed',lw=3,density=True)
plt.xlim([10000,2800])
plt.xlabel('Teff (K)')
plt.ylabel('Density')

plt.subplot(2,1,1)
plt.semilogy(tess_qlp['Effective Temperature Value'],tess_qlp['Star Radius Value'],'o',color='blue',label='QLP TOIs')
plt.semilogy(tess_spoc['Effective Temperature Value'],tess_spoc['Star Radius Value'],'o',color='black',label='SPOC TOIs')
plt.semilogy(kepler['koi_steff'],kepler['koi_srad'],'o',color='red',label='KOIs')
plt.xlim([10000,2800])
plt.ylim([0.1,20])
plt.ylabel('Radius (Solar)')
plt.legend()
plt.tight_layout()
plt.savefig('koi-toi-hrd.png',dpi=200)

ix=np.where(tess_spoc['Effective Temperature Value'] > 7000.)[0]
ascii.write(tess_spoc[ix],'tess_spoc_candidates.csv',delimiter=',')

ix=np.where(tess_qlp['Effective Temperature Value'] > 7000.)[0]
ascii.write(tess_qlp[ix],'tess_qlp_candidates.csv',delimiter=',')

