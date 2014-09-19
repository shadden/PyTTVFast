from ctypes import *
import numpy as np

DEFAULT_TRANSIT = -1
#LIBPATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"
LIBPATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"

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

class libwrapper(object):
	""" A wrapper class for the TTVFast C library.  Gives access to the function TTVFast """
	def __init__(self):
		
		self.lib = CDLL("%s/libttvfast.so" % LIBPATH)
		self._TTVFast = self.lib.TTVFast
		self._TTVFast.argtypes =[p1d,c_double,c_double,c_double,c_int,POINTER(CALCTRANSIT)\
		 , POINTER(CALCRV), c_int, c_int, c_int]
		self._TTVFast.restype = None
	def TTVFast(self,pars,dt,t0,tfinal,nplanets,CalcTransitsArray,CalcRVArray,nRV,n_events,input_flag):
		self._TTVFast(pars,dt,t0,tfinal,nplanets,CalcTransitsArray,CalcRVArray,nRV,n_events,input_flag)
#
class TTVCompute(object):
	def __init__(self):
		self.interface = libwrapper()

	def TransitTimes(self,tfin,planet_params,GM=1.0,t0=0.0,input_type='astro',dtfrac=0.025):
		"""Get transit times from a TTVFast N-body integration """
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
		else:
			vel2 = np.linalg.norm(planet_params[:,3:],axis=1)**2
			r	= np.linalg.norm(planet_params[:,:3],axis=1)**2
			GM_a = -2 * ( 0.5 * vel2 - GM /r )
			periods = 2 * pi * np.power( GM_a  , -1.5 ) * GM

		dt = dtfrac * np.min(periods)
		
		n_events = int(np.sum(np.ceil( (tfin-t0) / periods + 1) ))
		
		model = (n_events * CALCTRANSIT)()
		for transit in model:
			transit.time = DEFAULT_TRANSIT
			
		self.interface.TTVFast(params,dt,t0,tfin,nplanets,model,None,0,n_events, input_n)
		transits = np.array([ ( transit.planet,transit.time ) for transit in model if transit.time != DEFAULT_TRANSIT ])
		
		transitlists = []
		for i in range(nplanets):
			condition=transits[:,0]==i
			transitlists.append((transits[:,1])[condition])
		return transitlists
			
def linefit(x,y):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	const = np.ones(len(y))
	A = np.vstack([const,x]).T
	return np.linalg.lstsq(A,y)[0]

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
		pMax = max(self.period_estimates)
		pMin = min(self.period_estimates)
		# Begginings and ends of integration to be a little more than a full period of the longest period
		# planet 
		self.t0 = min([ np.min(times) for times in self.transit_times ]) - 0.1 * pMin
		self.tFin = max([ np.max(times) for times in self.transit_times ]) + 1.1 * pMax

	def CoplanarParametersTransits(self,params,**kwargs):
		"""\nReturn the transit times of a coplanar planet system for the given set of 'params'.
		'params' should be an N by 5 array, where N is the number of planets.  The parameters
		for each planet are: \n\t[Mass, e*cos(peri), e*sin(peri), Period, Mean Anomaly]
		Mean anomalies should be given in radians. """
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
		nbody_transits = self.TransitTimes(self.tFin, input, input_type='jacobi', t0 = self.t0 )
		return nbody_transits

	def CoplanarParametersFitness(self,params,**kwargs):
		"""\nReturn the chi-squared fitness of transit times of a coplanar planet system generatedf from
		 the given set of 'params'. 'params' should be an N by 5 array, where N is the number of planets.
		  The parameters for each planet are:
		\t [Mass, e*cos(peri), e*sin(peri), Period, Mean Anomaly]. 
		The fitness is returned as '-inf' if the N-body integration does not return a sufficient number of 
		transits to match the observed data for any of the planets."""
		nbody_transits = self.CoplanarParametersTransits(params)
		chi2 = 0.0

		for i in range(self.nplanets):
			observed_times = self.transit_times[i]
			observed_numbers = self.transit_numbers[i]
			uncertainties = self.transit_uncertainties[i]
			#
			nbody_times = (nbody_transits[i])[observed_numbers]
			if np.max(observed_numbers) > len(nbody_times):
				return -inf

			diff = (observed_times - nbody_times )			
			
			chi2+= -0.5 * np.sum( np.power(diff,2.) / np.power(uncertainties,2.) )

		return chi2
	def GenerateInitialConditions(self,masses,evectors):
		"""
		For a list of masses and eccentricity vectors, create inital condition parameters with
		periods set to the average planet periods measured by linear fit and with initial values
		of Mean Anomaly that result in the first observed transit occuring at the observed time
		for an unperturbed orbit.
		"""
		assert masses.shape[0]==evectors.shape[0],"Input parameters must be properly shaped numpy arrays!"
		assert masses.shape[1]==self.nplanets and evectors.shape[1]==self.nplanets ,"Input parameters must be properly shaped numpy arrays!"
		assert evectors.shape[2]==2,"Input parameters must be properly shaped numpy arrays!"
		
		periods = self.period_estimates
		initTransits = self.tInit_estimates
		epoch = self.t0
		params = []
		for initpar in zip(masses,evectors):
			mass = initpar[0]
			evecs = initpar[1]
			eccs = np.linalg.norm(evecs,axis=1)
			pmgs = np.arctan2(evecs[:,1],evecs[:,0])
			meanAnoms =  -1.*(initTransits - epoch) * 2 * np.pi / periods + 2. * evecs[:,1] - pmgs
			paramArray = np.array(np.vstack([mass,evecs[:,0],evecs[:,1],periods,meanAnoms]).T).reshape(-1)
			params.append(paramArray)
		return params

if __name__=="__main__":
	# planet 1
	mass=1.e-5
	per,e,i = 1.0, 0.02, 90.
	ArgPeri, MeanAnom = np.random.rand(2) * 2.0 * np.pi
	LongNode = 0.0
	els1 = np.array([mass,per,e,i,LongNode,ArgPeri,MeanAnom])
	els1[4:] *= 180. / np.pi
	pars1 = np.array([mass,e*np.cos(ArgPeri),e*np.sin(ArgPeri),per,MeanAnom])
	# planet 2
	mass=1.e-5
	per,e,i = 1.515, 0.03, 90.
	ArgPeri, MeanAnom = np.random.rand(2) * 2.0 * np.pi
	LongNode = 0.0
	els2 = np.array([mass,per,e,i,LongNode,ArgPeri,MeanAnom])
	els2[4:] *= 180. / np.pi
	pars2 = np.array([mass,e*np.cos(ArgPeri),e*np.sin(ArgPeri),per,MeanAnom])
	
	
	pars = np.array([pars1,pars2]).reshape(-1)
	#def TransitTimes(self,tfin,planet_params,GM=1.0,t0=0.0,input_type='astro',dtfrac=0.025):
	nbody_create = TTVCompute()
	transits1,transits2 = nbody_create.TransitTimes(100.,np.array([els1,els2]),input_type='jacobi')
	data1 = np.array([ (i,transit+np.random.normal(scale=1.e-4),1.e-4) for i,transit in enumerate(transits1) ])
	data2 = np.array([ (i,transit+np.random.normal(scale=1.e-4),1.e-4) for i,transit in enumerate(transits2) ])
	#data1 = np.loadtxt('./inner.ttv')	
	#data2 = np.loadtxt('./outer.ttv')
	observed_data = [data1,data2]
	nbody_fit = TTVFitness(observed_data)
	transits=nbody_fit.CoplanarParametersTransits(pars)
	
	masses = np.random.normal(2.e-5,1.e-6,(100,2))
	evecs = np.random.normal(0.0,0.01,(100,2,2))
	ics = nbody_fit.GenerateInitialConditions(masses,evecs)
	#print "inner planet:"
	#for i,time in enumerate(transits[0]):
	#	print i,time,data1[i%len(data1)][1]
	#for i,time in enumerate(transits[1]):
	#	print i,time,data2[i%len(data2)][1]
	#fit=nbody_fit.CoplanarParametersFitness(pars)
	#print "Fitness comptued to be %.3f"%fit
