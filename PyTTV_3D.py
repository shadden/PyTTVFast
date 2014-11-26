from ctypes import *
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize,curve_fit,leastsq

DEFAULT_TRANSIT = -1
import os
who =os.popen("whoami") 
if who.readline().strip() =='samuelhadden':
	print "On laptop..."
	LIBPATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"
else:
	print "On Quest..."
	LIBPATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"
who.close()

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
			print "Warning: TTVFast did not generate the expected number of transits!"
			print "Trying once more with smaller time step: ",
			try:
				self._TTVFast(pars,dt/3.,t0,tfinal,nplanets,CalcTransitsArray,CalcRVArray,nRV,n_events,input_flag)
				print "Succeeded"
				return True
			except RuntimeError:
				print "Failed"
				print "Parameters: ", " ".join( map(lambda x: "%.3g"%x, pars[2:]))
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
			return [],False
		
		transits = np.array([ ( transit.planet,transit.time ) for transit in model if transit.time != DEFAULT_TRANSIT ])
		if len(transits)==0:
			return [],False
		transitlists = []
		for i in range(nplanets):
			condition= transits[:,0] == i
			transitlists.append((transits[:,1])[condition])
		return transitlists,success
		
	def MCMC_Param_TransitTimes(self,planet_params,tFin):
		""" Return transit times for input parameters given in the form:
				[ mass, period, ex, ey, Ix, Iy , T_0 ]
			for each planet """
		mass,period,ex,ey,Ix,Iy,T0 = planet_params.reshape(-1, 7).T
		
		# Choose an epoch based on the first transit time
		epoch = np.min( T0 ) - 0.1 * min(period)
		
		# Convert to standard astrocentric orbels:
		
		ecc = np.sqrt( ex **2 + ey**2 )
		varpi = np.arctan2(ex,ey) + np.pi / 2.
		Inc = np.sqrt( Ix **2 + Iy**2 )
		Omega =  np.arctan2(Ix,Iy) + np.pi / 2.
		MeanLongitude = 2. * np.pi * ( epoch - T0 ) / period + 2 * ey + np.pi / 2.
		
		# 
		arg_peri = varpi - Omega
		MeanAnom = MeanLongitude - varpi
		
		# TTVFast Coordinates: [mass, period, e, i, Node, Peri, Mean Anomaly ]
		rad2deg = lambda x: np.mod(x *180. / np.pi ,360. )
		ttvfast_pars = np.vstack(( mass,period,ecc , rad2deg(Inc) , rad2deg(Omega) , rad2deg(arg_peri), rad2deg(MeanAnom))).T
		
		return self.TransitTimes(tfin,ttvfast_pars,t0=epoch)
		
class TransitObservations(object):
""" An object to store transit observations.""""
	def __init__(self,observed_transit_data):
		
		self.observed_transit_data = observed_transit_data
		self.nplanets = len(observed_transit_data)
		self.transit_numbers = [ (data[:,0]).astype(int) for data in observed_transit_data ] 
		self.transit_times = [ data[:,1] for data in observed_transit_data ] 
		self.transit_uncertainties = [ data[:,2] for data in observed_transit_data ] 
		
		self.flat_transits = []
		for transits in self.transit_times:
			for time in transits:
			flat_transits.append(time)
		self.flat_transits = np.array(transits)
			 
	def get_chi2(tansitsList):
		chi2 = 0.0
		for i,transits in enumerate(transitsList):
			chi2 += np.sum([  ((transits - self.transit_times[i]) / self.transit_uncertainties[i] )**2 ])
		return chi2
	
	def tFinal(self):
		return np.max( self.flat_transits )
	def tInit(self):
		return np.min( self.flat_transits )
		
class TTVFit(TTVCompute):
	def __init__(self,observed_transit_data):
		self.interface = libwrapper()
		self.Observations = TransitObservations(observed_transit_data)
	
	

if __name__=="__main__":
	test_elements = np.array([[ 1.e-5, 1.0, 0.03, 0.01, 0., 0. , 100.],\
							  [ 1.e-5, 2.05, 0.01, -0.01, 0., 0. , 101.]])
	
	nb = TTVCompute()
	nb.MCMC_Param_TransitTimes(test_elements,200.)
