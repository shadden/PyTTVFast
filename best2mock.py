import PyTTVFast as nTTV
import numpy as np
from scipy.optimize import curve_fit

def linefit2(x,y,sigma=None):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	linefn = lambda x,slope,intercept: x*slope + intercept
	return curve_fit(linefn,x,y,sigma=sigma)

if __name__=="__main__":
	# a comment

	def linefit_resids(x,y,sigma=None):
		s,m = linefit2(x,y,sigma)[0]
		return y - s*x -m
	#

	with open('planets.txt') as fi:
		planetNames = [l.strip() for l in fi.readlines()]

	nbody_fit = nTTV.TTVFitnessAdvanced([np.loadtxt(f) for f in planetNames])
	
	bestpars = np.loadtxt('bestpars.txt')
	best_ttimes,success = nbody_fit.CoplanarParametersTransformedTransits(bestpars)
	
	
	
