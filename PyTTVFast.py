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
		params = array([1.0,GM])
		for planet in planet_params:
			params = append(params,planet)
		
		if input_n==0 or input_n==1:
			periods = planet_params[:,1]
		else:
			vel2 = np.linalg.norm(planet_params[:,3:],axis=1)**2
			r	= np.linalg.norm(planet_params[:,:3],axis=1)**2
			GM_a = -2 * ( 0.5 * vel2 - GM /r )
			periods = 2 * pi * np.power( GM_a  , -1.5 ) * GM

		dt = dtfrac * np.min(periods)
		
		n_events = int(np.sum(ceil( (tfin-t0) / periods + 1) ))
		
		model = (n_events * CALCTRANSIT)()
		for transit in model:
			transit.time = DEFAULT_TRANSIT
			
		self.interface.TTVFast(params,dt,t0,tfin,nplanets,model,None,0,n_events, input_n)
		transits = array([ ( transit.planet,transit.time ) for transit in model if transit.time != DEFAULT_TRANSIT ])
		
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
	def __init__(self,observed_transit_data):
		self.observed_transit_data = observed_transit_data

		self.nplanets = len(observed_transit_data)
		self.transit_numbers = [ data[:,0] for data in observed_transit_data ] 
		self.transit_times = [ data[:,1] for data in observed_transit_data ] 
		self.transit_uncertainties = [ data[:,2] for data in observed_transit_data ] 
		
		self.period_estimates = [linefit(data[:,0],data[:,1])[1] for data in self.obesrved_transit_data ]
		# Begginings and ends of integration to be a little more than a full period of the longest period
		# planet 
		self.t0 = min([ np.min(times) for times in self.transit_times ]) - 1.1 * pMax
		self.tFin = max([ np.min(times) for times in self.transit_times ]) + 1.1 * pMax

	def CoplanarFitness(self,params,**kwargs):
		planet_params = params.reshape(-1,5)
		masses = planet_params[:,0]
		eccs = np.norm( planet_params[:,1:3] )
		arg_peri = np.arctan2(planet_params[:,2] , planet_params[:,1])
		periods = planet_params[:,3]
		meanAnoms = planet_params[:,4]
		incs = np.ones(self.nplanets)*90.
		Omegas = np.zeros(self.nplanets)
		
		input = np.vstack([masses,periods,eccs,incs,Omega,arg_peri,meanAnoms]).T
		nbody_transits = self.TransitTimes(self.tMax + X, input ,input_type='jacobi',t0 = self.tMin + Y , kwargs )

		chi2 = 0.0

		for i in range(self.nplanets):
			observed_times = self.transit_times[i]
			observed_numbers = self.transit_numbers[i]
			#
			if np.max(observed_numbers) > len(nbody_times):
				return -inf
			nbody_times = nbody_transits[i]
			uncertainties = self.transit_times[i]

			diff = (observed_times - nbody_times )			
			chi2 += -0.5 * np.sum( np.power(diff,2) / np.power(uncertainties,2) )

		return chi2
		

if False:
	# planet 1
	mass=1.e-5
	per,e,i = 1.0, 0.02, 90.
	LongNode, ArgPeri, MeanAnom = np.random.rand(3) * 360
	els1 = np.array([mass,per,e,i,LongNode,ArgPeri,MeanAnom])
	# planet 2
	mass=1.e-5
	per,e,i = 1.515, 0.03, 90.
	LongNode, ArgPeri, MeanAnom = np.random.rand(3) * 360
	els2 = np.array([mass,per,e,i,LongNode,ArgPeri,MeanAnom])
	
	nbody = TTVCompute()
	planet_params=np.array([els1,els2])
	transits = nbody.TransitTimes(100.,planet_params)
	
if __name__=="__main__":
	data1 = loadtxt('./inner.ttv')	
	data2 = loadtxt('./outer.ttv')
	observed_data = [data1,data2]
	nbody_fit = TTVFitness(observed_data)
	print nbody_fit.period_estimates
