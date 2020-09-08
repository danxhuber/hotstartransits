import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import pdb
from astroquery.mast import Catalogs 

# take the output of sample.py and collate information required to run isoclassify
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

# run isoscripts in commandline