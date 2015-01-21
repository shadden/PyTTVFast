import os
import PyTTV_3D as nTTV
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit

def linefit2(x,y,sigma=None):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	linefn = lambda x,slope,intercept: x*slope + intercept
	return curve_fit(linefn,x,y,sigma=sigma)

def linefit_resids(x,y,sigma=None):
	s,m = linefit2(x,y,sigma)[0]
	return y - s*x -m

def MakeArtificialDir(nbfit,pars,dir='./Artificial',showplot=False):
	tmax = np.max(map(np.max, nbfit.Observations.transit_times))
	transitTimes,sucess = nbfit.MCMC_CoplanarParam_TransitTimes(pars,tmax +3.)
	os.system('mkdir -p %s'%dir)
	fi = open("%s/planets.txt"%dir,"w")
	np.savetxt("%s/inpars.txt"%dir,pars)
	for i,times in enumerate(transitTimes):
		noiseLvl = np.median(nbfit.Observations.transit_uncertainties[i])
		nTransits = len(times)
		noise = np.random.normal(0.0,noiseLvl,nTransits	)
		noisyTimes = times + noise
		np.savetxt("%s/planet%d.txt"%(dir,i),np.vstack(( np.arange(nTransits) , noisyTimes, noiseLvl * np.ones(nTransits) ) ).T)
		fi.write("planet%d.txt\n"%i)
		p,t0  = linefit2( np.arange(nTransits) ,noisyTimes)[0]
		pl.errorbar(noisyTimes, noisyTimes - p* np.arange(nTransits) - t0 ,yerr=noiseLvl ,fmt='s') 

	pl.savefig('%s/Artificial_Transits.png'%dir)
	if showplot:
		pl.show()		

if  __name__=="__main__":
	
	# Open and read transit time files
	try:
		with open('planets.txt') as fi:
			planetNames = [l.strip() for l in fi.readlines()]
	except:
		raise Exception("Planets file(s) not found!")

	nbody_fit=nTTV.TTVFit([np.loadtxt(f) for f in planetNames])
	# Load best-fit parameters
if False:	
	try:
		bestpars = np.loadtxt('bestpars.txt')
	except:
		raise Exception("No file `bestpars.txt' found.")

	MakeArtificialDir(nbody_fit,bestpars,"Best",True)	
	
	try:
		chain,lnlike = np.loadtxt("chain.dat.gz"),np.loadtxt("chain.lnlike.dat.gz")
	except:
		raise Exception("Chain files not found or failed to load")

	bigmassI = np.argmax(np.sum(chain[lnlike>np.percentile(lnlike,90)][:,(0,3)],axis=1))
	MakeArtificialDir(nbody_fit,chain[bigmassI],"BigMass",True)
	
	
	littlemassI = np.argmin(np.sum(chain[lnlike>np.percentile(lnlike,90)][:,(0,3)],axis=1))
	MakeArtificialDir(nbody_fit,chain[littlemassI],"LittleMass",True)

	# Make a subdirectory for artificial transit times, make a plot of the artificial TTVs
	# 	and save the artificial data.

