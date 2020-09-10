import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

# analyze transit durations

# initial sample with transit parameters
spoc=ascii.read('tess_spoc_candidates.csv')
ids=['']*len(spoc)
for i in range(0,len(spoc)):
	ids[i]='TOI'+str(spoc['Full TOI ID'][i]).split('.')[0] 

# stellar parameters from isoclassify
iso_out=ascii.read('iso_output_spoc.csv')

# monte-carlo stuff to incorporate impact parameter and eccentricity distributions
ns=1000
gconst = 6.6726e-8
r_sun = 6.9599e10		
r_earth = 6.378e8

lhs=np.zeros(len(iso_out))
rp=np.zeros(len(iso_out))
periods=np.zeros(len(iso_out))
teffs=np.zeros(len(iso_out))
rstar=np.zeros(len(iso_out))
rhostar=np.zeros(len(iso_out))
tics=np.zeros(len(iso_out),dtype='int64')

for i in range(0,len(iso_out)):

	rhos=iso_out['iso_rho'][i]
	rhose=iso_out['iso_rho_err1'][i]
	rhoss=np.random.randn(ns)*rhose+rhos
	
	teffs[i]=iso_out['iso_teff'][i]
	rstar[i]=iso_out['iso_rad'][i]
	rhostar[i]=rhos
	
	# skip if isoclassify failed
	if (rhos == 0.):
		continue
	
	print(iso_out['id_starname'][i])
	print('rho_star:',rhos,rhose)

	um=np.where(np.asarray(ids) == iso_out['id_starname'][i])[0]

	td=np.float(spoc['Transit Duration Value'][um])
	tde=np.float(spoc['Transit Duration Error'][um])
	print('transit duration:',td,tde)
	
	tics[i]=spoc['TIC'][um]
	
	# hack for crazy transit durations
	if (tde/td > 0.1):
		tde=td*0.1
	tdur=np.random.randn(ns)*tde+td
	
	rprs=(spoc['Planet Radius Value'][um]*r_earth)/(spoc['Star Radius Value'][um]*r_sun)

	# sample impact parameters from 0 to 1-rp/rs (excludes grazing transits)
	ror=1.-rprs
	b=np.random.uniform(0.,ror,size=ns)
	ecc=np.random.rayleigh(0.05,size=ns)
	ecc=0. # assume circular for now
	#print('max b:',ror)

	period=spoc['Orbital Period Value'][um]

	arstar=(period/(np.pi*tdur/24.)) * np.sqrt(1-b**2)
	rhotransit = (3.*np.pi/(gconst*(period*86400.)**2))*arstar**3
	rhotransit=rhotransit*(1.-ecc**2)**(3./2.)
		
	#plt.ion()
	#plt.clf()
	#plt.hist(rhoss,bins=100,label='star')
	#plt.hist(rhotransit,bins=100,label='transit')
	#plt.xlabel('rho')
	#plt.title(iso_out['id_starname'][i])
	#plt.legend()
	#plt.savefig('plots/'+iso_out['id_starname'][i]+'.png',dpi=150)
	
		
	# calculate a hacked log likelihood to rank systems
	lh=(1./(rhose*np.sqrt(2.*np.pi)))*np.exp(-(((rhotransit)-rhos)**2)/(2.*rhose**2))
	print('log likelihood',np.log10(np.sum(lh)))
	
	lhs[i]=np.log10(np.sum(lh))
	rp[i]=rprs*iso_out['iso_rad'][i]*r_sun/r_earth
	periods[i]=period	
	#input(':')

# sort by loglh and output result
s=np.argsort(lhs)[::-1]
print('TOI TIC teff(K) rstar(rsun) rhostar(rhosun) period(d) rplanet(r_earth)')
for i in range(0,len(s)):
	if (teffs[s[i]] < 6500.):
		continue
	print(iso_out['id_starname'][s[i]],tics[s[i]],teffs[s[i]],rstar[s[i]],rhostar[s[i]],periods[s[i]],rp[s[i]])

