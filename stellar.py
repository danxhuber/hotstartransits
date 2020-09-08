import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import pdb
from astroquery.mast import Catalogs 

# take the sample and collate information required to run isoclassify
spoc=ascii.read('tess_spoc_candidates.csv')

ids=['']*len(spoc)
band=['kmag']*len(spoc)
dust=['allsky']*len(spoc)

for i in range(0,len(spoc)):
	ids[i]='TOI'+str(spoc['Full TOI ID'][i]).split('.')[0] 

catalogData = Catalogs.query_criteria(catalog = "Tic", ID = spoc['TIC'])

spoc=spoc[np.argsort(spoc['TIC'])]
catalogData=catalogData[np.argsort(np.asarray(catalogData['ID'],dtype='int'))]

ascii.write([ids,catalogData['ra'],catalogData['dec'],catalogData['gaiabp'],catalogData['e_gaiabp'],catalogData['gaiarp'],catalogData['e_gaiarp'],1./catalogData['d'],catalogData['e_d'] /catalogData['d']**2,catalogData['Kmag'],catalogData['e_Kmag'],band,dust,spoc['TIC']],'tess_spoc_candidates_isocl.csv',delimiter=',',names=['id_starname','ra','dec','bpmag','bpmag_err','rpmag','rpmag_err','parallax','parallax_err','kmag','kmag_err','band','dust','comment'])


# run isoclassify (see isoscripts)



# analyze transit durations
iso_out=ascii.read('iso_output_spoc.csv')
ns=1000
gconst = 6.6726e-8
r_sun = 6.9599e10		
r_earth = 6.378e8

lhs=np.zeros(len(iso_out))

for i in range(0,len(iso_out)):

	rhos=iso_out['iso_rho'][i]
	rhose=iso_out['iso_rho_err1'][i]
	rhoss=np.random.randn(ns)*rhose+rhos
	
	if (rhos == 0.):
		continue
	
	print(iso_out['id_starname'][i])
	print('rho_star:',rhos,rhose)

	um=np.where(np.asarray(ids) == iso_out['id_starname'][i])[0]

	td=np.float(spoc['Transit Duration Value'][um])
	tde=np.float(spoc['Transit Duration Error'][um])
	print('transit duration:',td,tde)
	
	if (tde/td > 0.1):
		tde=td*0.1
	tdur=np.random.randn(ns)*tde+td
	
	rprs=(spoc['Planet Radius Value'][um]*r_earth)/(spoc['Star Radius Value'][um]*r_sun)

	ror=1.-rprs
	b=np.random.uniform(0.,ror,size=ns)
	ecc=np.random.rayleigh(0.05,size=ns)
	ecc=0.
	#print('max b:',ror)

	period=spoc['Orbital Period Value'][um]

	arstar=(period/(np.pi*tdur/24.)) * np.sqrt(1-b**2)
	rhotransit = (3.*np.pi/(gconst*(period*86400.)**2))*arstar**3
	rhotransit=rhotransit*(1.-ecc**2)**(3./2.)
		
	plt.ion()
	plt.clf()
	plt.hist(rhoss,bins=100,label='star')
	plt.hist(rhotransit,bins=100,label='transit')
	plt.xlabel('rho')
	plt.title(iso_out['id_starname'][i])
	plt.savefig('plots/'+iso_out['id_starname'][i]+'.png',dpi=150)
		
	lh=(1./(rhose*np.sqrt(2.*np.pi)))*np.exp(-(((rhotransit)-rhos)**2)/(2.*rhose**2))
	print('log likelihood',np.log10(np.sum(lh)))
	
	lhs[i]=np.log10(np.sum(lh))
	
	#input(':')


s=np.argsort(lhs)[::-1]
for i in range(0,len(s)):
	print(iso_out['id_starname'][s[i]],lhs[s[i]])

