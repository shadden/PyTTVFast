import PyTTVFast as nTTV
import numpy as np
from scipy.optimize import curve_fit

def linefit2(x,y,sigma=None):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	linefn = lambda x,slope,intercept: x*slope + intercept
	return curve_fit(linefn,x,y,sigma=sigma)

def linefit_resids(x,y,sigma=None):
	s,m = linefit2(x,y,sigma)[0]
	return y - s*x -m

if __name__=="__main__":
	
	# Open and read transit time files
	try:
		with open('planets.txt') as fi:
			planetNames = [l.strip() for l in fi.readlines()]
		nbody_fit=TTVFitnessAdvanced([np.loadtxt(f) for f in planetNames])

	except:
		raise Exception("Planets file(s) not found!")

	# Load best-fit parameters
	
	try:
		transitTimes,sucess = nbody_fit.CoplanarParametersTransformedTransits(np.loadtxt('bestpars.txt'))
	except:
		raise Exception("No file `bestpars.txt' found.")

		
	
	# Make a subdirectory for artificial transit times, make a plot of the artificial TTVs
	# 	and save the artificial data.

	os.system('mkdir -p ./Artificial')
	fi = open("Artificial/planets.txt","w")
	for i,times in enumerate(transitTimes):
		noiseLvl = median(nbody_fit.transit_uncertainties[i])
		nTransits = len(times)
		noise = np.random.normal(0.0,noiseLvl,nTransits	)
		noisyTimes = times + noise
		np.savetxt("Artificial/planet%d.txt"%i,vstack(( arange(nTransits) , noisyTimes, noiseLvl * np.ones(nTransits) ) ).T)
		fi.write("planet%d.txt\n"%i)
		t0,p  = linefit( np.arange(nTransits) ,noisyTimes)
		errorbar(noisyTimes, noisyTimes - p* arange(nTransits) - t0 ,yerr=noiseLvl ,fmt='s') 

	savefig('Artificial/Artificial_Transits.png')
		
	show()		