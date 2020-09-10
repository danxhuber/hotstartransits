import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
from exoplanet.gp import terms, GP
from exoplanet import estimate_inverse_gamma_parameters


#np.random.seed(123)
#periods = np.random.uniform(5, 20, 2)
#t0s = periods * np.random.rand(2)
#t = np.arange(0, 80, 0.02)
#yerr = 5e-4

dat=ascii.read('toi2143.csv')
t=np.array(dat['time'])
y=np.array(dat['flux'])
y=y-np.mean(y)

yerr = np.zeros(len(y))+0.001


um=np.where((t > 1986.5) & (t < 1987))[0]
plt.plot(t[um],y[um])

t=t[um]
y=y[um]

with pm.Model() as model:

    mean = pm.Normal("mean", mu=0.0, sigma=1.0)
    S1 = pm.InverseGamma(
        "S1", **estimate_inverse_gamma_parameters(0.5 ** 2, 10.0 ** 2)
    )
    S2 = pm.InverseGamma(
        "S2", **estimate_inverse_gamma_parameters(0.25 ** 2, 1.0 ** 2)
    )
    w1 = pm.InverseGamma(
        "w1", **estimate_inverse_gamma_parameters(2 * np.pi / 10.0, np.pi)
    )
    w2 = pm.InverseGamma(
        "w2", **estimate_inverse_gamma_parameters(0.5 * np.pi, 2 * np.pi)
    )
    log_Q = pm.Uniform("log_Q", lower=np.log(2), upper=np.log(10))

    # Set up the kernel an GP
    kernel = terms.SHOTerm(S_tot=S1, w0=w1, Q=1.0 / np.sqrt(2))
    kernel += terms.SHOTerm(S_tot=S2, w0=w2, log_Q=log_Q)
    gp = GP(kernel, t, yerr ** 2, mean=mean)

    # Condition the GP on the observations and add the marginal likelihood
    # to the model
    gp.marginal("gp", observed=y)
  
pm.summary(trace, varnames=["S1", "S2", "w1", "w2"])
  
  
with model:
    map_soln = xo.optimize(start=model.test_point)
    
with model:
    mu, var = xo.eval_in_model(
        gp.predict(t, return_var=True, predict_mean=True), map_soln
    )
    
plt.ion()
plt.clf()
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")
# Plot the prediction and the 1-sigma uncertainty
sd = np.sqrt(var)
art = plt.fill_between(t, mu + sd, mu - sd, color="C1", alpha=0.3)
art.set_edgecolor("none")
plt.plot(t, mu, color="C1", label="prediction")




