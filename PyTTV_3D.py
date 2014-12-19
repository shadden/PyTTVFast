#	Normal coords	|	TTVFast coords
#---------------------------------------
#	theta = 0		| 	theta' = pi /2
#	ex =e*cos(w)	|	ex' = e*cos(w' + pi/2) = -ey 	
#	ey				|	ey' = ex
		

def linefit(x,y):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	const = np.ones(len(y))
	A = np.vstack([const,x]).T
	return np.linalg.lstsq(A,y)[0]
def line_resids(x,y):
	y0,s = linefit(x,y)
	return y - s*x - y0
	
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
	
	def TransitData(self,tfin,planet_params,GM=1.0,t0=0.0,input_type='astro',dtfrac=0.02):
		"""
		Get transit times as well as the sky position and velocity from a TTVFast N-body integration.
		
		Parameters
		----------
		tfin : The stop-time of the integration
		
		planet_params: an N-planet by 7 array, with each entry in the form:
			[mass, period, e, i, Node, Peri, Mean Anomaly ]
			
		GM : Gravitational constant time stellar mass
		
		t0 : Start time of integration, default = 0
		
		input_type : The input coordinate type (astro, jacobi, or cartesian)
		
		Returns
		-------
		 transit data: A list containing arrays for the transit of each planet.
		 	Each array conatains the transit time, sky-position and sky-velocity.
		 
		 success: A boolean variable that indicates whether the integration succeeded or not.
		"""
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
		
		transitdata = np.array([ ( transit.planet,transit.time, transit.rsky, transit.vsky) for transit in model if transit.time != DEFAULT_TRANSIT ])
		
		if len(transits)==0:
			return [],False
		transitlists = []
		for i in range(nplanets):
			condition= transitdata[:,0] == i
			transitlists.append((transitdata[:,1:])[condition])
		return transitlists,success
	
	def MCMC_Params_To_TTVFast(self,planet_params):
		""" Return TTVFast coordinate inputs for input parameters given in the form:
				[ mass, period, ex, ey, I, Omega , T_0 ]
			for each planet """
		mass,period,ex,ey,I,Omega,T0 = planet_params.reshape(-1, 7).T
		
		# Choose an epoch based on the first transit time
		epoch = np.min( T0 ) - 0.1 * min(period)
		
		# Convert to standard astrocentric orbels:
		#------------------------------------------
		ecc = np.sqrt( ex **2 + ey**2 )
		arg_peri = np.arctan2(ey,ex)  
		
		MeanLongitude = 2. * np.pi * ( epoch - T0 ) / period  + 0.5 * np.pi + Omega
		varpi = arg_peri + Omega 
		MeanAnom = MeanLongitude - varpi
		
		# TTVFast Coordinates: [mass, period, e, i, Node, Argument Peri(?), Mean Anomaly ]
		rad2deg = lambda x: np.mod(x * 180. / np.pi ,360. )
		ttvfast_pars = np.vstack(( mass,period,ecc , rad2deg(I) , rad2deg(Omega) , rad2deg(arg_peri), rad2deg(MeanAnom))).T

		return epoch,ttvfast_pars
	
	def MCMC_Param_TransitTimes(self,planet_params,tFin,full_data=False):
		""" Return transit times for input parameters given in the form:
				[ mass, period, ex, ey, I, Omega , T_0 ]
			for each planet """
		epoch,ttvfast_pars = self.MCMC_Params_To_TTVFast(planet_params)
		if full_data:
			return self.TransitData(tFin,ttvfast_pars,t0=epoch)
			
		return self.TransitTimes(tFin,ttvfast_pars,t0=epoch)
	
	def MCMC_CoplanarParam_TransitTimes(self,coplanar_planet_params,tFin,full_data=False):
		""" Return transit times for input parameters given in the form:
				[ mass, period, ex, ey, I, Omega , T_0 ]
			for each planet """
		mass,period,ex,ey,T0 = coplanar_planet_params.reshape(-1, 5).T
		npl = len(mass)
		full_pars = np.vstack((mass,period,ex,ey,np.pi/2.*np.ones(npl),np.zeros(npl),T0)).T			
		return self.MCMC_Param_TransitTimes(full_pars,tFin,full_data=full_data)
	
	def MCMC_RelativeNodeParam_TransitTimes(self,rel_node_planet_params,tFin,full_data=False):
		""" Return transit times for input parameters given in the form:
				1st planet: 	[ mass, period, ex, ey, I, T_0 ] (Omega = 0)
				Other planets:	[ mass, period, ex, ey, I, Omega=0 , T_0 ]
			for each planet """
		m0,p0,ex0,ey0,I0,T00 = rel_node_planet_params[:6]
		p0params = np.array([m0,p0,ex0,ey0,I0,0.0,T00])
		full_pars = np.vstack(( p0params,rel_node_planet_params[6:].reshape(-1, 7) ))
		return self.MCMC_Param_TransitTimes(full_pars,tFin,full_data=full_data)

##########################################################################################		
#
#			Transit Observtaions Class
#			--------------------------
#			
#			Represents a collection of observed transit times for a multi-planet system.
#
#			Methods:
#			--------
#			get_chi2(transitList): 
#				Return the resulting chi^2 value from comparing `transitList'  to observed
#				 transits.
#
#			tFinal():
#				Final transit observed.
#
#			tInit():
#				First transit observed.
##########################################################################################

class TransitObservations(object): 
	def __init__(self,observed_transit_data):
		self.observed_transit_data = observed_transit_data
		self.nplanets = len(observed_transit_data)
		self.transit_numbers = [ (data[:,0]).astype(int) for data in observed_transit_data ] 
		self.transit_times = [ data[:,1] for data in observed_transit_data ] 
		self.transit_uncertainties = [ data[:,2] for data in observed_transit_data ] 
		
		self.flat_transits = []
		for transits in self.transit_times:
			for time in transits:
				self.flat_transits.append(time)
		self.flat_transits = np.array(self.flat_transits)
		
		self.PeriodEstimates = []
		self.tInitEstimates = []
		for nums,times in zip(self.transit_numbers,self.transit_times):
			self.PeriodEstimates.append( linefit(nums,times)[1] )
			self.tInitEstimates.append( linefit(nums,times)[0] )

		self.PeriodEstimates = np.array( self.PeriodEstimates )
		self.tInitEstimates = np.array( self.tInitEstimates )
			 
	def get_chi2(self,transitsList):
		chi2 = 0.0
		for i,transits in enumerate(transitsList):
			chi2 += np.sum([  ((transits - self.transit_times[i]) / self.transit_uncertainties[i] )**2 ])
		return chi2
	
	def tFinal(self):
		return np.max( self.flat_transits )
	def tInit(self):
		return np.min( self.flat_transits )
	
class ImpactParameterObservations(object):
		r2au = 0.004649
		def __init__(self,rstar_data,mstar_data,b_data):
			"""	
			Store data on stellar mass/radius uncertainty and inclinations through
			impact parameters.
			"""
			self.Rstar,self.Rstar_err = rstar_data
			self.Mstar,self.Mstar_err = mstar_data
			# impact parameters
			self.b,self.b_err = b_data.T
		
		def ImpactParametersToInclinations(self,periods):
			""" 
			Convert periods and impact parameters to inclinations using stellar radius and mass.
			""" 	
 			rstar = self.r2au * self.Rstar
			sigma_rstar = self.r2au * self.Rstar_err
			mstar = self.Mstar
			sigma_mstar = self.Mstar_err
			b = self.b
			sigma_b = self.b_err
			
			a = ( mstar * (periods/ 365.25)**2 )**(1./3.)	
			cosi0 = np.abs( self.b * rstar / a )
	
			#sigma_cosi ~  Sum [ d (cosi) /dq * sigma_q ]
			dcos_dM = -(1./3.) * cosi0  / mstar
			dcos_dr = b / a
			dcos_db = rstar / a
 			sigma_cosi = np.sqrt( (dcos_dr * sigma_rstar)**2 + (dcos_dM * sigma_mstar )**2 + (dcos_db * sigma_b )**2 )

 			return np.arccos(cosi0) , np.diag(sigma_cosi)
 		
 		def ImpactParametersPriors(self,inclinations, periods):
			""" 
			Gaussian priors based on formula (a/R_*) cos(i) = b.  Assumes that fractional uncertainties
			are sufficiently small that we can linearize cos(i)'s dependence on the parameter errors
			and therefore add error contributions in quadrature.
			"""
			rstar = self.r2au * self.Rstar
			sigma_rstar = self.r2au * self.Rstar_err
			mstar = self.Mstar
			sigma_mstar = self.Mstar_err
			b = self.b
			sigma_b = self.b_err
	
			cosi = np.abs( np.cos(inclinations) ) 
			a = ( mstar * (periods/ 365.25)**2 )**(1./3.)
	
			cosi0 = np.abs( b * rstar / a )
	
			#sigma_cosi ~  Sum [ d (cosi) /dq * sigma_q ]
			dcos_dM = -(1./3.) * cosi0  / mstar
			dcos_dr = b / a
			dcos_db = rstar / a
	
 			sigma_cosi = np.sqrt( (dcos_dr * sigma_rstar)**2 + (dcos_dM * sigma_mstar )**2 + (dcos_db * sigma_b )**2 )

 			return -0.5 * np.sum ( (cosi - cosi0)**2 / sigma_cosi**2 )

##########################################################################################		
#
#			TTVFit Class
#			--------------------------
#			
#			Combine a collection of observed transit times for a multi-planet system
#			with methods to compute the likelihood of TTVs generated by N-body integration
#
#			Initialize as `TTVFit(observed_data)'
#
#			Methods:
#			--------
#			ParameterFitness(planet_params): 
#				Return the resulting chi^2 value of transits generated from 
#				 the input planet parameters.
#
#			tFinal():
#				Final transit observed.
#
#			tInit():
#				First transit observed.
##########################################################################################

			


class TTVFit(TTVCompute):
	def __init__(self,observed_transit_data):
		self.interface = libwrapper()
		self.Observations = TransitObservations(observed_transit_data)
	
	def ParameterFitness(self,planet_params):
		"""
		Return log-likelihood = -0.5* chi*2 computed from planet parameters given 
		as a list in the form:
			mass, period, ex, ey , I , Omega, T0
				---- OR ----
			[	[mass0, period0, ex0, ey0 , I0 , T00],
				[mass1, period1, ex1, ey1 , I1 , dOmega, T01],
				...
			]
				---- OR ----
			mass, period, ex, ey , T0
		for each planet.
		"""
		
		tFinal = self.Observations.tFinal() + np.max(self.Observations.PeriodEstimates)
		if planet_params.shape[-1]%7==0:
			transits,success = self.MCMC_Param_TransitTimes(planet_params,tFinal)
		elif len(planet_params.reshape(-1)) == 7 * self.Observations.nplanets - 1:
			transits,success = self.MCMC_RelativeNodeParam_TransitTimes(planet_params,tFinal)
		elif planet_params.shape[-1]%5==0:
			transits,success = self.MCMC_CoplanarParam_TransitTimes(planet_params,tFinal)
		else:
			print "Bad input dimensions!"
			raise ValueError()
		
		if not success:
			return -np.inf
		try:
			transit_list = [ transits[i][nums] for i,nums in enumerate(self.Observations.transit_numbers) ]
		except:
			# If the number of computed transits is less than the number of observed transits
			#	then -infinity should be returned
			return -np.inf
			
			
		chi2 = self.Observations.get_chi2(transit_list)
		return - 0.5 * chi2	
		
	def LeastSquareParametersFit(self,params0,inclination_data=None):
		"""
		Use L-M minimization to find the best-fit set of input parameters along with an estimated covariance matrix.
		The method will assume coplanar orbits or full 3D orbits based on the number of input parameters.
		"""
		npl = self.Observations.nplanets
		if len(params0.reshape(-1)) == npl * 5:
			coplanar = True
		elif len(params0.reshape(-1)) == npl * 7:
			coplanar = False
		else:
			print "Shape of initial parameter does not match what is required for the number of planets!"
			raise
			
		target_data = np.array([])
		errors = np.array([])
		
		for time,err in zip(self.Observations.transit_times,self.Observations.transit_uncertainties):
			target_data = np.append(target_data,time)
			errors = np.append(errors,err)
		
		tFinal = self.Observations.tFinal() + np.max(self.Observations.PeriodEstimates)
		
		def objectivefn(x):
			
			if coplanar:
				transits,success = self.MCMC_CoplanarParam_TransitTimes(x,tFinal)
			else:
				transits,success = self.MCMC_Param_TransitTimes(x,tFinal)
			if	inclination_data:
					assert not coplanar, "Inclination data should not be include for coplanar fits"
					cosi = np.abs( np.cos( x.reshape(-1,7)[:,4] ) )
					cosi0 = inclination_data[0]
					cosi_err = inclination_data[0]
					inc_chi2 = (cosi - cosi0) / cosi_err
			
			answer = np.array([],dtype=float)
			for i,t in enumerate(transits):
				tnums = self.Observations.transit_numbers[i]
				try:
					answer = np.append( answer,np.array(t[tnums]) )
				except:
					return -np.inf * np.ones(len(target_data))
			#
			try:
				ttvchi2 = (answer - target_data)/errors
			except:
				return -np.inf * np.ones(len(target_data))
			
			if inclination_data:
				return np.append(ttvchi2,inc_chi2)
			else:
				return ttvchi2
		
		return leastsq(objectivefn, params0,full_output=1)
	
	
	def ParameterPlot(self,planet_params,ShowObs=True):
		"""
		Plot TTVs from planet parameters given as a list in the form:
			mass, period, ex, ey , I , Omega, T0
				--- OR ---
			mass, period, ex, ey , T0
		for each planet.
		"""
		
		tFinal = self.Observations.tFinal() + 0.1 * np.min(self.Observations.PeriodEstimates)
		if planet_params.shape[-1]%7==0:
			transits,success = self.MCMC_Param_TransitTimes(planet_params,tFinal)
		elif planet_params.shape[-1]%5==0:
			transits,success = self.MCMC_CoplanarParam_TransitTimes(planet_params,tFinal)
		else:
			print "Bad input dimensions!"
			raise
		assert success, "Failed to generate TTVs from specified parameters!"

		npl = self.Observations.nplanets
		T0 = self.Observations.tInitEstimates

		color_pallette = ['b','r','g']
		axList = []
		for i in range(npl):
			if i==0:
				axList.append( pl.subplot( 100 *npl + 10 + i + 1) )
			else:
				axList.append( pl.subplot(100*npl + 10 + i +1,sharex=axList[0]) )
			if i != npl-1:
					pl.setp( axList[i].get_xticklabels(), visible=False )
					
			col = color_pallette[i%len(color_pallette)]
			per = self.Observations.PeriodEstimates[i]		
			pl.plot(transits[i], transits[i] - np.arange(len(transits[i])) * per - T0[i],"%s-"%col)
	
			if ShowObs:
				otimes = self.Observations.transit_times[i]
				ebs = self.Observations.transit_uncertainties[i]
				trNums = self.Observations.transit_numbers[i]
				pl.errorbar(otimes, otimes - trNums * per - T0[i],yerr=ebs,color=col,fmt='s')
	
			# Re-label y-ticks in minutes
			locs,labls = pl.yticks()
			locs = map(lambda x: int(round(24*60*x))/(24.*60.),locs[1:-1])
			pl.yticks( locs , 24.*60.*np.array(locs) ) 
	
		pl.subplots_adjust(hspace=0.0)
		
		

		
	def coplanar_initial_conditions(self,mass,ex,ey):
		npl = self.Observations.nplanets
		assert mass.shape[0]==ex.shape[0]==ey.shape[0]==npl, "Improper input dimensions!"
		return np.vstack((mass,self.Observations.PeriodEstimates,ex,ey,np.ones(npl)*np.pi/2.,np.zeros(npl),self.Observations.tInitEstimates)).T
			
	
#########################################################################################################
#########################	Run fitting of observed transits and inclinations	#########################
#########################################################################################################
if False:

	nwalkers = 200

	infile = 'planets.txt'
	
	with open(infile,'r') as fi:
		infiles = [ line.strip() for line in fi.readlines()]
	
	input_data =[]
	for file in infiles:
		input_data.append( loadtxt(file) )
	nplanets = len(input_data)


	while min( array([ min(tr[:,0]) for tr in input_data])  ) != 0:
		print "re-numbering transits..."
		for data in input_data:
			data[:,0] -= 1
	
	nbody_fit = TTVFit(input_data)
	npl = nbody_fit.Observations.nplanets
	
	with open("inclination_data.txt") as fi:
		lines = [l.split() for l in fi.readlines()]
	mstar,sigma_mstar = map(float,lines[0])
	rstar,sigma_rstar = map(float,lines[1])
	b,sigma_b = np.array([map(float,l[1:]) for l in lines[2:] ]).T	
	b_Obs = ImpactParameterObservations([rstar,sigma_rstar],[mstar,sigma_mstar], vstack((b,sigma_b)).T)
	
	pars0=nbody_fit.coplanar_initial_conditions(1.e-5*ones(3),random.normal(0,0.02,3),random.normal(0,0.02,3) )
	cp_pars0=pars0[:,(0,1,2,3,6)]
	
	fitdata = nbody_fit.LeastSquareParametersFit(cp_pars0)
	print "Fitness: %.2f"%nbody_fit.ParameterFitness(fitdata[0])
	
	best,cov = fitdata[:2]
	
	best3d = best.reshape(-1,5)
	i0,sigma_i=b_Obs.ImpactParametersToInclinations(nbody_fit.Observations.PeriodEstimates)
	best3d = np.hstack(( best3d[:,:4], i0.reshape(-1,1) , np.random.uniform(-0.005,0.005,(npl,1)), best3d[:,-1].reshape(-1,1) ))
	print best3d.shape
	best3d = best3d.reshape(-1)
	fitdata = nbody_fit.LeastSquareParametersFit( best3d )
	best,cov = fitdata[:2]
	
	
	fitdata2 = nbody_fit.LeastSquareParametersFit( best ,inclination_data = [cos(i0),diagonal(sigma_i)] )
	best,cov = fitdata2[:2]
	print "3D Fitness: %.2f"%nbody_fit.ParameterFitness(best)
	print "3D likelihood: %.2f"%fit(best)
	p = random.multivariate_normal(best,cov/25.,size=nwalkers)

	    		

	
#########################################################################################################
#########################				Fit generated data 						#########################
#########################################################################################################
	
if __name__=="__main__":
	import sys
	tfin = 70 * 45.1
	pratio = 1.53
	b1 = 0.251
	b2 = 0.
	with open("./00_test_directory/inclination_data.txt","w") as fi:
		fi.write("1.0    0.12\n")
		fi.write("1.0    0.2\n")
		fi.write("b       0.151   0.138\n")
		fi.write("c       .85    0.141\n")
	
	with open("./00_test_directory/inclination_data.txt") as fi:
		lines = [l.split() for l in fi.readlines()]
	mstar,sigma_mstar = map(float,lines[0])
	rstar,sigma_rstar = map(float,lines[1])
	b,sigma_b = np.array([map(float,l[1:]) for l in lines[2:] ]).T
	b_Obs = ImpactParameterObservations([rstar,sigma_rstar],[mstar,sigma_mstar], vstack((b,sigma_b)).T)
	inc,sigma_inc= b_Obs.ImpactParametersToInclinations(np.array([45.1 , pratio*45.1]))
	
	test_elements = np.array([[ 1.e-5, 1.0*45.1, 0.1, 0.06,  inc[0] , 0. , 100.],\
							  [ 1.e-5, pratio*45.1,0.03,-0.07, inc[1] , -0.03 , 106.]])
	
	savetxt('./00_test_directory/true_parameters.txt',test_elements)
	
	nb = TTVCompute()
	transits,success = nb.MCMC_Param_TransitTimes(test_elements,tfin )
	obs_data = []
	noise_lvl = 2.5e-4 * 45.1
	for times in transits:
		ntimes = len(times)
		noise  = np.random.normal(0,noise_lvl,ntimes)
		obs_data.append( np.vstack((np.arange(ntimes),times+noise, np.ones(ntimes)*noise_lvl )).T   )
	
	fit = TTVFit(obs_data)
	p0 = test_elements.copy().reshape(-1)
	print 
	dOmega = 0.3
	p0[5] += dOmega
	p0[5+7] += dOmega
	p1 = p0.copy()
	p1[5+7]-=  p1[5]
	p1[5]  -=  p1[5]
	p1=np.hstack((p1[:5] ,p1[6:] ))
	print "First transit time"
	print transits[0][0], "\n",nb.MCMC_Param_TransitTimes(p0,tfin)[0][0][0],"\n"
	
	print "Parameter Fitness"
	print fit.ParameterFitness(test_elements.reshape(-1)),"\n",fit.ParameterFitness(p0),"\n",fit.ParameterFitness(p1),"\n"
	
	print "TTV Fast Coordinates"
	for i in range(2):
		print " ".join(map(lambda x: "%.4f"%x,fit.MCMC_Params_To_TTVFast(test_elements)[1][i]))
		print
	
	for i,ttimes in enumerate(obs_data):
		np.savetxt("./00_test_directory/planet%d.txt"%i,ttimes)
	
	fit.ParameterPlot(test_elements)
	pl.show()
