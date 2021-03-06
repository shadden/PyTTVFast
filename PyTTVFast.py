from ctypes import *
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize,curve_fit,leastsq

DEFAULT_TRANSIT = -1
import os
who =os.popen("whoami") 
if who.readline().strip() =='samuelhadden':
	print("On laptop...")
	LIBPATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"
else:
	print("On Quest...")
	LIBPATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"
who.close()
LIBPATH = "/home/shadden/Projects/03_TESS_TTV/PyTTVFast"

PLOTS = False

def get_ctype_ptr(dtype,dim,**kwargs):
	return np.ctypeslib.ndpointer(dtype=dtype,ndim=dim,flags='C',**kwargs)
	
p1d=get_ctype_ptr(np.float,1)
p2d = get_ctype_ptr(np.float,2)
p0 = POINTER(c_double)
p1dInt = get_ctype_ptr(np.int,1)

class CALCTRANSIT(Structure):
	_fields_=[("planet",c_int),("epoch",c_int),\
			  ("time",c_double),("rsky",c_double),\
			  ("vsky",c_double)]
class CALCRV(Structure):
	_fields_=[("time",c_double),("RV",c_double)]
	
"""	Form of TTVFast function:
	void TTVFast(double *params,double dt, 
				double Time, double total,
				int n_plan,CalcTransit *transit,
				CalcRV *RV_struct, int nRV,
				 int n_events, int input_flag)
***************************************************
The arguments for TTVFast are:
TTVFast(parameter array, time step, t0, tfinal, number of planets, 
	structure to hold calculated transit information, 
	structure to hold calculated RV information, 
	number of RV observations, size of transit array, input style flag)
***************************************************
Depending on what type of input you want to give 
(the options are Jacobi elements, Astrocentric elements, Astrocentric cartesian) 
the form of the input parameter array will be either
For elements: 
	****************
G Mstar Mplanet Period E I LongNode Argument Mean Anomaly (at t=t0, the reference time)....
repeated for each planet
: 2+nplanets*7 in length
	****************
or, for Cartesian:
	****************
G Mstar Mplanet X Y Z VX VY VZ (at t=t0, the reference time)....
repeated for each planet
: 2+nplanets*7 in length
*******************************
G is in units of AU^3/day^2/M_sun, all masses are in units of M_sun, 
the Period is in units of days, and the angles are in DEGREES. 
Cartesian coordinates are in AU and AU/day. 
The coordinate system is as described in the paper text. One can use different units, as long as G is converted accordingly. 
"""
def PeriodConversion(params):
	m1,ex1,ey1,p1,M1 = params[:5]
	m2,ex2,ey2,p2,M2 = params[5:]
	
	L1 = M1 + np.arctan2(ey1,ex1)
	L2 = M2 + np.arctan2(ey2,ex2)
	
	j = np.argmin([abs((k-1.) * p2 / (k * p1) - 1.)  for k in range(2,5)]) + 2
	Lj = j*L2 - (j-1)*L1
	Delta = (j-1.) * p2 / (j * p1) - 1.
	assert j==2 or j==3, "Need to enter Laplace coefficients for j=%d"%j
	if j==2:
		f = -1.19049
		f1 = 0.42839  
	elif j==3:
		f = -2.02522
		f1 = 2.48401 
	
	Zcos = (f*ex1 + f1 * ex2 )*np.cos(Lj) + (f*ey1 + f1*ey2)*np.sin(Lj)
	
	da1_a1 =  m2 *(j-1) * 2 * Zcos / (j * Delta * (p1/p2)**(1./3.) )
	da2_a2 = -1. * m1 * 2 * Zcos / ( Delta )
	
	return np.array((m1,ex1,ey1,p1*(1-1.5*da1_a1),M1,\
				  m2,ex2,ey2,p2*(1-1.5*da2_a2),M2 ))

class libwrapper(object):
	""" A wrapper class for the TTVFast C library.  Gives access to the function TTVFast """
	def __init__(self):
		
		self.lib = CDLL("%s/libttvfast.so" % LIBPATH)
		self._TTVFast = self.lib.TTVFast
		self._TTVFast.argtypes =[p1d,c_double,c_double,c_double,c_int,POINTER(CALCTRANSIT)\
		 , POINTER(CALCRV), c_int, c_int, c_int]
		self._TTVFast.restype = c_int
		def check_errors(ret, func, args):
			if ret<0:
				raise RuntimeError("TTVFast returned error code %d for the given arguments"%ret)
			return ret
		self._TTVFast.errcheck = check_errors
	def TTVFast(self,pars,dt,t0,tfinal,nplanets,CalcTransitsArray,CalcRVArray,nRV,n_events,input_flag):
		try:
			self._TTVFast(pars,dt,t0,tfinal,nplanets,CalcTransitsArray,CalcRVArray,nRV,n_events,input_flag)
			return True
		except RuntimeError:
			print( "Warning: TTVFast did not generate the expected number of transits!")
			print( "Trying once more with smaller time step: ")
			try:
				self._TTVFast(pars,dt/3.,t0,tfinal,nplanets,CalcTransitsArray,CalcRVArray,nRV,n_events,input_flag)
				print( "Succeeded")
				return True
			except RuntimeError:
				print("Failed")
				print(("Parameters: ", " ".join( ["%.3g"%x for x in pars[2:]])))
				return False
#
class TTVCompute(object):
	def __init__(self):
		self.interface = libwrapper()

	def TransitTimes(self,tfin,planet_params,GM=1.0,t0=0.0,input_type='astro',dtfrac=0.02):
		"""Get transit times from a TTVFast N-body integration.
		:param planet_params: an N-planet by 7 array, with each entry in the form:
			[mass, period, e, i, Node, Peri, Mean Anomaly ]"""
		input_types = ['jacobi','astro','cartesian']
		assert input_type in  input_types, "Invalid input type, valid choised are: '%s', '%s', or '%s'"%(input_types[0],input_types[1],input_types[2])
		input_n = input_types.index(input_type)
			
		assert type(planet_params)==np.ndarray and (planet_params.shape)[-1]==7, "Bad planet parameters array!"

		nplanets=len(planet_params)
		params = np.array([1.0,GM])
		for planet in planet_params:
			params = np.append(params,planet)
		
		if input_n==0 or input_n==1:
			periods = planet_params[:,1]
			eccs = planet_params[:,2]
		else:
			vel2 = np.linalg.norm(planet_params[:,3:],axis=1)**2
			r	= np.linalg.norm(planet_params[:,:3],axis=1)**2
			GM_a = -2 * ( 0.5 * vel2 - GM /r )
			periods = 2 * pi * np.power( GM_a  , -1.5 ) * GM

			# this should be fixed!
			eccs = np.zeros(nplanets)
			
		# Don't let the periapse set timestep less than 1/10th the planet period
		dtfactors =  np.maximum( np.power( (1. - eccs ) ,1.5) , 0.1 )
		dt = dtfrac * np.min( periods * dtfactors )
		
		n_events = int(np.sum(np.ceil( (tfin-t0) / periods + 1) )) + 1
		
		model = (n_events * CALCTRANSIT)()
		for transit in model:
			transit.time = DEFAULT_TRANSIT
		
		success = self.interface.TTVFast(params,dt,t0,tfin,nplanets,model,None,0,n_events, input_n)
		
		if not success:
		    print("Internal TTVFast error.")
		    return [],False
		
		transits = np.array([ ( transit.planet,transit.time ) for transit in model if transit.time != DEFAULT_TRANSIT ])
		if len(transits)==0:
			print("No transits recorded.")
			return [],False
		transitlists = []
		for i in range(nplanets):
			condition= transits[:,0] == i
			transitlists.append((transits[:,1])[condition])
		return transitlists,success
			
def linefit(x,y):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	const = np.ones(len(y))
	A = np.vstack([const,x]).T
	return np.linalg.lstsq(A,y)[0]
def residplot(data):
	n = np.arange(len(data))
	t0,p= linefit(n, data)
	pl.figure()
	pl.plot(data, data - p * n - t0)
	pl.show()
	return t0,p
	
class TTVFitness(TTVCompute):
	"""
	A class for evaluating the chi-squared fitness of sets of orbital parameters in reproducing a set of observed transit times.
	An TTVFitness object is instantiated with a list of the observed transit data.  The transit data for each planet should be in 
	the form of Nx3 array, with each entry having the form:
	\t\t\t[ transit number, transit time, uncertainty in transit time].
	"""
	def __init__(self,observed_transit_data):
		self.interface = libwrapper()

		self.observed_transit_data = observed_transit_data

		self.nplanets = len(observed_transit_data)
		self.transit_numbers = [ (data[:,0]).astype(int) for data in observed_transit_data ] 
		self.transit_times = [ data[:,1] for data in observed_transit_data ] 
		self.transit_uncertainties = [ data[:,2] for data in observed_transit_data ] 
		
		self.tInit_estimates,self.period_estimates = np.array([linefit(data[:,0],data[:,1]) for data in self.observed_transit_data ]).T
		#self.transitOrder = np.argsort( np.array([tTimes[0] for tTimes in self.transit_times]) )
		self.transitOrder = np.argsort( self.tInit_estimates )
		pMax = max(self.period_estimates)
		pMin = min(self.period_estimates)
		# Begginings and ends of integration to be a little more than a full period of the longest period
		# planet 
		self.t0 = min([ np.min(times) for times in self.transit_times ]) - 0.1 * pMin
		self.tFin = max([ np.max(times) for times in self.transit_times ]) + 3.1 * pMax # extra padding if a bad period is guessed...

	def CoplanarParametersTransits(self,params,**kwargs):
		"""\nReturn the transit times of a coplanar planet system for the given set of 'params'.
		'params' should be an N by 5 array, where N is the number of planets.  The parameters
		for each planet are: \n\t[Mass, e*cos(peri), e*sin(peri), Period, Mean Anomaly]
		Mean anomalies should be given in radians. """
		t0 = kwargs.get('t0',self.t0)
		planet_params = params.reshape(-1,5)
		masses = planet_params[:,0]
		eccs = np.linalg.norm( planet_params[:,1:3],axis=1 )
		arg_peri = np.arctan2(planet_params[:,2] , planet_params[:,1]) * 180. / np.pi
		periods = planet_params[:,3]
		meanAnoms = planet_params[:,4] * 180. / np.pi
		# Coplanar planets -- all inclinations are 90 deg. (relative to plane of sky)
		incs = np.ones(self.nplanets)*90.
		Omegas = np.zeros(self.nplanets)
		
		input = np.vstack([masses,periods,eccs,incs,Omegas,arg_peri,meanAnoms]).T
		nbody_transits,success = self.TransitTimes(self.tFin, input, input_type='jacobi', t0 = t0 )
		return nbody_transits,success

	def CoplanarParametersFitness(self,params,convert_periods=False):
		"""\nReturn the chi-squared fitness of transit times of a coplanar planet system generated from
		 the given set of 'params'. 'params' should be an N by 5 array, where N is the number of planets.
		  The parameters for each planet are:
		\t [Mass, e*cos(peri), e*sin(peri), Period, Mean Anomaly]. 
		The fitness is returned as '-inf' if the N-body integration does not return a sufficient number of 
		transits to match the observed data for any of the planets."""
		
		if convert_periods:
			assert self.nplanets==2, "Period conversion currently only supported for 2 planets!"
			params = PeriodConversion(params)
		nbody_transits,success = self.CoplanarParametersTransits(params)
		if not success:
			return -np.inf
		chi2 = 0.0

		for i in range(self.nplanets):
			observed_times = self.transit_times[i]
			observed_numbers = self.transit_numbers[i]
			uncertainties = self.transit_uncertainties[i]
			#
			if np.max(observed_numbers) > len(nbody_transits[i]):
				return -np.inf
			nbody_times = (nbody_transits[i])[observed_numbers]

			diff = (observed_times - nbody_times )			
			
			chi2+= -0.5 * np.sum( np.power(diff,2.) / np.power(uncertainties,2.) )

		return chi2
	
	def FitBestPeriods(self,masses,evecs):
		def f(periods):
			pmgs = np.arctan2(evecs[:,1],evecs[:,0])
			# TTVFast coordinates have observer along z-axis so planets transit when theta = pi/2
			meanAnoms =  0.5 * np.pi - (self.tInit_estimates - self.t0) * 2 * np.pi / periods - 2. * evecs[:,0] - pmgs
			meanAnoms = np.mod(meanAnoms,2.*np.pi)
			pars = np.array(np.vstack([masses,evecs[:,0],evecs[:,1],periods,meanAnoms]).T).reshape(-1)
			#return pars
			return -1.0 * self.CoplanarParametersFitness(pars)

		ptol = 1.e-2
		period_bounds = [np.array([1.-ptol,1.+ptol ])*p for p in self.period_estimates]	
		res = minimize(f,self.period_estimates,bounds=period_bounds,method='L-BFGS-B') #'TNC' 
		return res
			
	def GenerateInitialConditions(self,masses,evectors,lazy=True,style=1):
		"""
		For a list of masses and eccentricity vectors, create inital condition parameters with
		periods set to the average planet periods measured by linear fit and with initial values
		of Mean Anomaly that result in the first observed transit occuring at the observed time
		for an unperturbed orbit.
		"""
		assert masses.shape[0]==evectors.shape[0],"Input parameters must be properly shaped numpy arrays!"
		assert masses.shape[1]==self.nplanets and evectors.shape[1]==self.nplanets ,"Input parameters must be properly shaped numpy arrays!"
		assert evectors.shape[2]==2,"Input parameters must be properly shaped numpy arrays!"
		
		initTransits = self.tInit_estimates
		epoch = self.t0
		params = []
		for initpar in zip(masses,evectors):
			mass = initpar[0]
			evecs = initpar[1]
			if lazy:
				periods = self.period_estimates
			else:
				periods = self.FitBestPeriods(mass,evecs).x
			pmgs = np.arctan2(evecs[:,1],evecs[:,0])
			# TTVFast coordinates have observer along z-axis so planets transit when theta = pi/2
			#meanLongs = 2.*np.pi * (nbody_fit.tInit_estimates[0]- 0.25*nbody_fit.period_estimates[0] - nbody_fit.tInit_estimates) / nbody_fit.period_estimates + 0.5 * np.pi
		
			meanAnoms =   0.5 * np.pi - (initTransits - epoch) * 2 * np.pi / periods - 2. * evecs[:,0] - pmgs
			meanAnoms = np.mod(meanAnoms + np.pi,2. * np.pi) - np.pi
		
			ic = np.array(np.vstack([mass,evecs[:,0],evecs[:,1],periods,meanAnoms]).T).reshape(-1)
			params.append(ic)
		if style == 1:
			return params
		
	def GenerateRandomInitialConditions(self,masses,sigma_dm_m,evecs,sigma_e,N,**kwargs):
		
		mass_list = masses * (1. + sigma_dm_m * np.random.randn(N,self.nplanets) )
		evecs_list = evecs  +  sigma_e * np.random.randn(N,self.nplanets,2)
		iclist = np.array(self.GenerateInitialConditions(mass_list,evecs_list,lazy=True))
		
		periods = iclist.reshape(N,-1,5)[:,:,3] * (1.+1.e-4*np.random.randn(N,self.nplanets)) 
		period_indices = tuple([3+5*n for n in range(self.nplanets)])
		iclist[:,period_indices] = periods
		return iclist

	def PlotTTV(self,params):
		transits,success = self.CoplanarParametersTransits(params)
		otransits = self.transit_times
		pl.figure()
		color_pallette = ['b','r','g']
		for i in range(self.nplanets):
			col = color_pallette[i%len(color_pallette)]
			per = self.period_estimates[i]
			tInit = self.tInit_estimates[i]
			pl.plot(transits[i],transits[i] - per * np.arange(len(transits[i])) - tInit,'%so'%col) 
			pl.errorbar(otransits[i],otransits[i] - per * self.transit_numbers[i] - tInit,fmt='%ss'%col,yerr=self.transit_uncertainties[i])

class TTVFitnessAdvanced(TTVFitness):
	"""
	A class for evaluating the chi-squared fitness of sets of orbital parameters in reproducing a set of observed transit times.
	TTVFitnessAdvanced overwrites the 'CoplanarParametersFitness' of the TTVFitness class by re-parameterizing periods and Mean
	Anomolies.
	"""	
	def __init__(self,observed_transit_data):
		super(TTVFitnessAdvanced, self).__init__(observed_transit_data)
		self.t0 = 0.0
		self.pratios = self.period_estimates / np.min(self.period_estimates)
		self.tFin = np.max([ np.max(times) for times in self.transit_times ]) / np.min(self.period_estimates) + 1.1 * max(self.pratios) 
		
	
	def GenerateInitialConditions(self,masses,evectors,lazy=True,style=1):
		"""
		For a list of masses and eccentricity vectors, create inital condition parameters with
		periods set to the average planet periods measured by linear fit and with initial values
		of Mean Anomaly that result in the first observed transit occuring at the observed time
		for an unperturbed orbit.
		"""
	#	assert masses.shape[0]==evectors.shape[0],"Input parameters must be properly shaped numpy arrays!"
	#	assert masses.shape[1]==self.nplanets and evectors.shape[1]==self.nplanets ,"Input parameters must be properly shaped numpy arrays!"
	#	assert evectors.shape[2]==2,"Input parameters must be properly shaped numpy arrays!"
		
		params = []
		if self.transitOrder[0]==0:
			meanLongs = 2.*np.pi * (self.tInit_estimates[0] - 0.25*self.period_estimates[0] - self.tInit_estimates) / self.period_estimates + 0.5 * np.pi
			meanLongs = np.mod(meanLongs[1:] + np.pi,2. * np.pi)-np.pi
		elif np.min(self.tInit_estimates) > self.transit_times[0][0] - 0.25 * self.period_estimates[0]:
			meanLongs = 2.*np.pi * (self.tInit_estimates[0] - 0.25*self.period_estimates[0] - self.tInit_estimates) / self.period_estimates + 0.5 * np.pi
			meanLongs = np.mod(meanLongs[1:] + np.pi,2. * np.pi)-np.pi	
		else:

			meanLongs = 2.*np.pi * (self.tInit_estimates[0] - 1.25*self.period_estimates[0] - self.tInit_estimates) / self.period_estimates + 0.5 * np.pi
			meanLongs = np.mod(meanLongs[1:] + np.pi,2. * np.pi)-np.pi
			
		periods = self.period_estimates[1:]/self.period_estimates[0]
		
		params = np.append( np.hstack((masses.reshape(-1,1),evectors)) , np.array( list(zip(periods,meanLongs)) ) )
		return params
		
	def GenerateRandomInitialConditions(self,masses,sigma_dm_m,evecs,sigma_e,N,**kwargs):
		
		mass_list = masses * (1. + sigma_dm_m * np.random.randn(N,self.nplanets) )
		evecs_list = evecs  +  sigma_e * np.random.randn(N,self.nplanets,2)
		iclist = []
		for mass,evec in zip(mass_list,evecs_list):
			iclist.append(self.GenerateInitialConditions(mass,evec))
		return iclist

	def CoplanarParametersFitness(self,params):
		"""
		Return the chi-squared fitness of transit times of a coplanar planet system generated from
		 the given set of 'params'. 'params' should be an (Nx5)-2 array, where N is the number of planets.
		 Once generated, the transit times will be transformed using linear least-squares to solve for a
		 rescaling factor and time offset.  This allows periods to be represented relative to the period 
		 of the innermost planet and initial angular positions expressed relative to the initial angular 
		 position of the innermost planet, thus reducing the free parameters by 2 when model-fitting.
		"""
		masses_and_evecs = params[:3*self.nplanets].reshape(-1,3)
		mass = masses_and_evecs[:,0]
		ex,ey = masses_and_evecs[:,1],masses_and_evecs[:,2]
		pmgs = np.arctan2(ey,ex)
		
		other_params = params[3*self.nplanets:].reshape(-1,2)
		periodRatios = np.append(np.array([1.0]), other_params[:,0])
		meanLongs = np.append(np.array([0.0]), other_params[:,1])		

		nbody_params = (np.vstack([ mass,ex,ey,periodRatios,meanLongs-pmgs ]).T).reshape(-1)

		nbody_transits,success = self.CoplanarParametersTransits(nbody_params,t0=0.0)
		if not success:
			return -np.inf
		
		####################################################################################################################
		nbodyTransitOrder = np.argsort( np.array([ ntransits[0] for ntransits in nbody_transits] ))
		while nbodyTransitOrder[0] != self.transitOrder[0]:
			firstToTransit = self.transitOrder[0]
			for i in np.arange(nbodyTransitOrder.tolist().index(firstToTransit)):
				planetNumber = nbodyTransitOrder[i]
				nbody_transits[planetNumber] = nbody_transits[planetNumber][1:]
			
			nbodyTransitOrder = np.argsort( np.array([ntransits[0] for ntransits in nbody_transits]) )	
		####################################################################################################################
			
		observed_times,observed_numbers,uncertainties,nbody_times = np.array([]),np.array([]),np.array([]),np.array([])
		
		for i in range(self.nplanets):
			if max(self.transit_numbers[i]) >= len(nbody_transits[i]):
				return -np.inf

			observed_times=np.append(observed_times, self.transit_times[i])
			observed_numbers=np.append(observed_numbers, self.transit_numbers[i])
			uncertainties=np.append(uncertainties,self.transit_uncertainties[i])
			nbody_times=np.append(nbody_times , (nbody_transits[i])[self.transit_numbers[i]])
		
		
		def func(x,tau,t0):
			return tau * x + t0

		# Solve for the transform of N-body time that gives the best fit to observed transits
		x0 = np.array((self.period_estimates[0],self.tInit_estimates[0]))
		tau,t0 = curve_fit(func, nbody_times, observed_times,x0, uncertainties)[0]
		
		transform = np.vectorize(lambda x: func(x,tau,t0))
		
		chi2 = 0.0
		
		for i in range(self.nplanets):
			uncertainties = self.transit_uncertainties[i]
			diff = transform((nbody_transits[i])[self.transit_numbers[i]]) - self.transit_times[i]
			chi2+= -0.5 * np.sum( np.power(diff,2.) / np.power(uncertainties,2.) )

		return chi2
		
	def CoplanarParametersTransformedTransits(self,params,observed_only=False):
		"""
		Return the transit times of a coplanar planet system generated from
		 the given set of 'params'. 'params' should be an (Nx5)-2 array, where N is the number of planets.
		 Once generated, the transit times will be transformed using linear least-squares to solve for a
		 rescaling factor and time offset.  This allows periods to be represented relative to the period 
		 of the innermost planet and initial angular positions expressed relative to the initial angular 
		 position of the innermost planet, thus reducing the free parameters by 2 when model-fitting.
		"""
		masses_and_evecs = params[:3*self.nplanets].reshape(-1,3)
		mass = masses_and_evecs[:,0]
		ex,ey = masses_and_evecs[:,1],masses_and_evecs[:,2]
		pmgs = np.arctan2(ey,ex)
		
		other_params = params[3*self.nplanets:].reshape(-1,2)
		periodRatios = np.append(np.array([1.0]), other_params[:,0])
		meanLongs = np.append(np.array([0.0]), other_params[:,1])		

		nbody_params = (np.vstack([ mass,ex,ey,periodRatios,meanLongs-pmgs ]).T).reshape(-1)

		nbody_transits,success = self.CoplanarParametersTransits(nbody_params,t0=0.0)

		if not success:
			return np.array([]), success
		
		####################################################################################################################
		nbodyTransitOrder = np.argsort( np.array([ ntransits[0] for ntransits in nbody_transits] ))
		while nbodyTransitOrder[0] != self.transitOrder[0]:
			firstToTransit = self.transitOrder[0]
			for i in np.arange(nbodyTransitOrder.tolist().index(firstToTransit)):
				planetNumber = nbodyTransitOrder[i]
				nbody_transits[planetNumber] = nbody_transits[planetNumber][1:]
			
			nbodyTransitOrder = np.argsort( np.array([ntransits[0] for ntransits in nbody_transits]) )	
		####################################################################################################################
	
		observed_times,observed_numbers,uncertainties,nbody_times = np.array([]),np.array([]),np.array([]),np.array([])	
		for i in range(self.nplanets):
			if max(self.transit_numbers[i]) >= len(nbody_transits[i]):
				return np.array([]),False

			observed_times=np.append(observed_times, self.transit_times[i])
			observed_numbers=np.append(observed_numbers, self.transit_numbers[i])
			uncertainties=np.append(uncertainties,self.transit_uncertainties[i])
			nbody_times=np.append(nbody_times , (nbody_transits[i])[self.transit_numbers[i]])
		
		def func(x,tau,t0):
			return tau * x + t0

		# Solve for the transform of N-body time that gives the best fit to observed transits
		x0 = np.array((self.period_estimates[0],self.tInit_estimates[0]))
		tau,t0 = curve_fit(func, nbody_times, observed_times,x0, uncertainties)[0]
		
		transform = np.vectorize(lambda x: func(x,tau,t0))
		if observed_only:
			soln=[]
			for i in range(self.nplanets):
				nums = self.transit_numbers[i]
				soln.append(transform(nbody_transits[i])[nums])
			
			return soln,True
		else:
			return [ transform(x) for x in (nbody_transits)],True
		
	def convert_params(self,params):
		""" 
		Convert old parameter format to new parameters (expressed relative to innermost planet)
		"""
		shaped_params = params.reshape(-1,5)
		pars = shaped_params[:,(0,1,2)].reshape(-1)
		
		period = shaped_params[1:,3]/shaped_params[0,3] 
		meanLong = np.mod(shaped_params[1:,4] - shaped_params[0,4]/period + np.pi,2*np.pi) - np.pi	
		
		return np.append(pars, np.vstack((period,meanLong)).T.reshape(-1) )
		
	def CoplanarParametersTTVPlot(self,params):
		"""Plot the TTVs of a set of input parameters against the observed TTVs"""
		transits,success = self.CoplanarParametersTransformedTransits(params)
		assert success, "Failed to generate TTVs from specified parameters!"
		otransits = self.transit_times
		pl.figure()
		color_pallette = ['b','r','g']
		
		for i in range(self.nplanets):
			plnum  = 100 * self.nplanets + 10 + (i+1)
			pl.subplot(plnum)
			col = color_pallette[i%len(color_pallette)]
			per = self.period_estimates[i]
			tInit = self.tInit_estimates[i]
			pl.plot(transits[i],transits[i] - per * np.arange(len(transits[i])) - tInit,'%so'%col) 
			pl.errorbar(otransits[i],otransits[i] - per * self.transit_numbers[i] - tInit,fmt='%ss'%col,yerr=self.transit_uncertainties[i])
		return transits

	def CoplanarParametersTTVResidPlot(self,params):
		"""Plot the TTVs of a set of input parameters against the observed TTVs"""
		transits,success = self.CoplanarParametersTransformedTransits(params)
		assert success, "Failed to generate TTVs from specified parameters!"
		otransits = self.transit_times
		pl.figure()
		color_pallette = ['b','r','g']
		
		for i in range(self.nplanets):
			plnum  = 100 * self.nplanets + 10 + (i+1)
			pl.subplot(plnum)
			col = color_pallette[i%len(color_pallette)]
			per = self.period_estimates[i]
			tInit = self.tInit_estimates[i]
			#pl.plot(transits[i],transits[i] - per * np.arange(len(transits[i])) - tInit,'%so'%col) 
			ntrans = self.transit_numbers[i]
			pl.errorbar(otransits[i],otransits[i] - transits[i][ntrans],fmt='%ss'%col,yerr=self.transit_uncertainties[i])
		return transits

	def CoplanarParametersTTVFit(self,params0):
		"""Use L-M minimization to find the best-fit set of input parameters along with an estimated covariance matrix"""
		target_data = np.array([])
		errors = np.array([])
		#for t in self.transit_times:
		for i in range(self.nplanets):
			target_data = np.append(target_data,self.transit_times[i])
			errors = np.append(errors,self.transit_uncertainties[i])
			
		def objectivefn(x):
			
			times =  self.CoplanarParametersTransformedTransits(x,observed_only=True)[0]
			answer = np.array([],dtype=float)
			for t in times:
				answer = np.append( answer,np.array(t) )
			#
	 
			try:
				return (answer - target_data)/errors
			except:
				return -np.inf * np.ones(len(target_data))
		
		return leastsq(objectivefn, params0,full_output=1)
		
	def PeriodAndLongitudeFit(self,params0):
		"""
		Use L-M minimization to find the best-fit set of periods and mean longitudes for a fixed set of masses and eccentricities
		along with an estimated covariance matrix"""
		target_data = np.array([])
		errors = np.array([])
		#for t in self.transit_times:
		for i in range(self.nplanets):
			target_data = np.append(target_data,self.transit_times[i])
			errors = np.append(errors,self.transit_uncertainties[i])
			
		massAndEcc = params0[:self.nplanets*3]
		periodAndLong0 = params0[self.nplanets*3:]
		def objectivefn(x):
			inpt = np.hstack((massAndEcc,x))
			times =  self.CoplanarParametersTransformedTransits(inpt,observed_only=True)[0]
			answer = np.array([],dtype=float)
			for t in times:
				answer = np.append( answer,np.array(t) )
			#
	 
			return (answer - target_data)/errors
		
		best,cov =  leastsq(objectivefn, periodAndLong0 ,full_output=1)[:2]
		return np.hstack((massAndEcc,best)),cov
		
if __name__=="__main__":
	# a comment
	def linefit2(x,y,sigma=None):
	
		assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
		linefn = lambda x,slope,intercept: x*slope + intercept
		return curve_fit(linefn,x,y,sigma=sigma)
	def linefit_resids(x,y,sigma=None):
		s,m = linefit2(x,y,sigma)[0]
		return y - s*x -m
	#
	try:
		with open('planets.txt') as fi:
			planetNames = [l.strip() for l in fi.readlines()]
		nbody_fit=TTVFitnessAdvanced([np.loadtxt(f) for f in planetNames])
	except:
		raise Exception("Planets file(s) not found!")


	try:
		transitTimes,sucess = nbody_fit.CoplanarParametersTransformedTransits(np.loadtxt('bestpars.txt'))
	except:
		raise Exception("No file `bestpars.txt' found.")

		
	
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
				
