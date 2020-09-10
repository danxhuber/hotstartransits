import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import lightkurve as lk


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

lcf=lk.search_lightcurvefile("TIC103195323").download()

lcpdc=lcf.PDCSAP_FLUX

period=5.589

ph=lcpdc.time % period

plt.ion()
plt.clf()
plt.plot(ph,lcpdc.flux)

um=np.where((ph > 0.12) & (ph < 5.4))[0]
plt.plot(ph[um],lcpdc[um].flux)

lcpdc2=lcpdc[um]
plt.clf()
plt.plot(lcpdc2.time,lcpdc2.flux)

p=lcpdc2.remove_nans().to_periodogram()

plt.ion()
plt.clf()
plt.plot(p.frequency/0.0864,p.power)
plt.xlim([100,600])
plt.xlabel('freq (muHz)')
plt.ylabel('power')
plt.tight_layout()

ascii.write([p.frequency/0.0864,p.power],'toi2143-ps.csv',delimiter=',',names=['freq','power'])
ascii.write([lcpdc2.time,lcpdc2.flux],'toi2143-ts.csv',delimiter=',',names=['time','flux'])

plt.savefig('toi2143-ps.png',dpi=150)


plt.ion()
plt.clf()
plt.plot(lcpdc.time,lcpdc.flux)

ascii.write([lcpdc.time,lcpdc.flux/np.median(lcpdc.flux)],'toi2143.csv',delimiter=',',names=['time','flux'])
