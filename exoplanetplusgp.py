import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import theano.tensor as tt
from exoplanet.gp import terms, GP
from exoplanet import estimate_inverse_gamma_parameters

dat=ascii.read('toi2143.csv')
t=np.array(dat['time'])
y=np.array(dat['flux'])
y=y-np.mean(y)
yerr = 0.001
periods=5.589
t0s=1984.07

ph = t % periods

um=np.where((ph < 0.2) | (ph > 5.3))[0]
plt.plot(ph[um],y[um])

plt.ion()
plt.clf()
plt.plot(t[um],y[um],'.')

t=t[um]
y=y[um]
yerr = np.zeros(len(y))+0.001

orbit = xo.orbits.KeplerianOrbit(period=periods,t0=t0s,b=0.5,rho_star=0.14,r_star=2.7)
u = [0.3, 0.2]
light_curve = (
    xo.LimbDarkLightCurve(u)
    .get_light_curve(orbit=orbit, r=0.19, t=t).eval())

plt.clf()
plt.plot(t,y,'.')
plt.plot(t, light_curve, color='red', lw=2)
plt.xlim([1994.8,1995.6])

def build_model(mask=None, start=None):

	with pm.Model() as model:

		# The baseline flux
		mean = pm.Normal("mean", mu=0.0, sd=0.00001)

		# The time of a reference transit for each planet
		t0 = pm.Normal("t0", mu=t0s, sd=1.0, shape=1)

		# The log period; also tracking the period itself
		logP = pm.Normal("logP", mu=np.log(periods), sd=0.01, shape=1)

		rho_star = pm.Normal("rho_star", mu=0.14, sd=0.01, shape=1)
		r_star = pm.Normal("r_star", mu=2.7, sd=0.01, shape=1)

		period = pm.Deterministic("period", pm.math.exp(logP))

		# The Kipping (2013) parameterization for quadratic limb darkening paramters
		u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))

		r = pm.Uniform(
			"r", lower=0.01, upper=0.3, shape=1, testval=0.15)
	
		b = xo.distributions.ImpactParameter(
			"b", ror=r, shape=1, testval=0.5)
	
		# Transit jitter & GP parameters
		logs2 = pm.Normal("logs2", mu=np.log(np.var(y)), sd=10)
		logw0 = pm.Normal("logw0", mu=0, sd=10)
		logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y)), sd=10)

		# Set up a Keplerian orbit for the planets
		orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, rho_star=rho_star,r_star=r_star)
	
		# Compute the model light curve using starry
		light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
			orbit=orbit, r=r, t=t
		)
		light_curve = pm.math.sum(light_curves, axis=-1) + mean

		# Here we track the value of the model light curve for plotting
		# purposes
		pm.Deterministic("light_curves", light_curves)

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

		gp.marginal("gp", observed=y)
		pm.Deterministic("gp_pred", gp.predict())

		# The likelihood function assuming known Gaussian uncertainty
		pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)

		# Fit for the maximum a posteriori parameters given the simuated
		# dataset
		map_soln = xo.optimize(start=model.test_point)
	
		return model, map_soln
	
model, map_soln = build_model()

gp_mod = map_soln["gp_pred"] + map_soln["mean"]
plt.clf()
plt.plot(t, y, ".k", ms=4, label="data")
plt.plot(t, gp_mod, lw=1,label="gp model")
plt.plot(t, map_soln["light_curves"], lw=1,label="transit model")
plt.xlim(t.min(), t.max())
plt.ylabel("relative flux")
plt.xlabel("time [days]")
plt.legend(fontsize=10)
_ = plt.title("map model")

np.random.seed(42)
with model:
    trace = pm.sample(
        tune=3000,
        draws=3000,
        start=map_soln,
        cores=2,
        chains=2,
        step=xo.get_dense_nuts_step(target_accept=0.9),
    )
    
    
pm.summary(trace, varnames=["period", "t0", "r", "b", "u", "mean", "rho_star","S1","S2", "w1", "w2"])


import corner

samples = pm.trace_to_dataframe(trace, varnames=["period", "r", "b", "rho_star"])

_ = corner.corner(
    samples,
    labels=["period", "r", "b", "rho_star"],
)




# Compute the GP prediction
gp_mod = np.median(trace["gp_pred"] + trace["mean"][:, None], axis=0)
t_mod = np.median(trace["light_curves"][:,:,0] + trace["mean"][:, None], axis=0)

gponly=gp_mod-t_mod

# Get the posterior median orbital parameters
p = np.median(trace["period"])
t0 = np.median(trace["t0"])


plt.clf()

plt.subplot(2,1,1)
plt.plot(t, y, ".k", ms=4, label="data")
plt.plot(t, gp_mod, lw=1,label="gp model")
plt.plot(t, t_mod,label="tr model",lw=2,color='red')
plt.xlim(t.min(), t.max())
plt.ylabel("relative flux")
plt.xlabel("time [days]")
plt.legend(fontsize=10)
plt.xlim([1995.04,1995.44])
plt.ylim([-0.0125,0.0075])
plt.title('TOI-2143')

plt.subplot(2,1,2)
ph = t % p
s=np.argsort(ph)
plt.plot(ph,y-gponly, ".k", ms=4, label="data")
plt.plot(ph[s], t_mod[s],label="tr model",lw=2,color='red')
plt.ylabel("relative flux")
plt.xlabel("phase [days]")
plt.legend(fontsize=10)
plt.xlim([5.2,5.6])
plt.ylim([-0.0125,0.0075])
plt.tight_layout()
plt.savefig('toi2143_gpmodel.png',dpi=150)

# Plot the folded data
x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
plt.plot(x_fold, y - gp_mod, ".k", label="data", zorder=-1000)

# Overplot the phase binned light curve
bins = np.linspace(-0.41, 0.41, 50)
denom, _ = np.histogram(x_fold, bins)
num, _ = np.histogram(x_fold, bins, weights=y)
denom[num == 0] = 1.0
plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned")

# Plot the folded model
inds = np.argsort(x_fold)
inds = inds[np.abs(x_fold)[inds] < 0.3]
pred = trace["light_curves"][:, inds, 0]
pred = np.percentile(pred, [16, 50, 84], axis=0)
plt.plot(x_fold[inds], pred[1], color="C1", label="model")
art = plt.fill_between(
    x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5, zorder=1000
)
art.set_edgecolor("none")

# Annotate the plot with the planet's period
txt = "period = {0:.5f} +/- {1:.5f} d".format(
    np.mean(trace["period"]), np.std(trace["period"])
)
plt.annotate(
    txt,
    (0, 0),
    xycoords="axes fraction",
    xytext=(5, 5),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=12,
)

plt.legend(fontsize=10, loc=4)
plt.xlim(-0.5 * p, 0.5 * p)
plt.xlabel("time since transit [days]")
plt.ylabel("de-trended flux")
plt.xlim(-0.15, 0.15);