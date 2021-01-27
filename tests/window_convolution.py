import functools
import logging
import numpy
import scipy
from scipy import special,constants,integrate
import math
from fractions import Fraction
from pywindow import WindowFunction,utils

def G(p):
	"""Return the function G(p), as defined in Wilson et al 2015.
	See also: WA Al-Salam 1953
	Taken from https://github.com/nickhand/pyRSD.

	Returns
	-------
	numer, denom: int
		the numerator and denominator

	"""
	toret = 1
	for p in range(1,p+1): toret *= (2*p - 1)
	return	toret,math.factorial(p)

def coefficients(ellout,ellin,front_coeff=True,as_string=False):
	
	coeffs = []
	qvals = []
	retstr = []
	
	for p in range(min(ellin,ellout)+1):

		numer = []
		denom = []

		# numerator of product of G(x)
		for r in [G(ellout-p), G(p), G(ellin-p)]:
			numer.append(r[0])
			denom.append(r[1])

		# divide by this
		a,b = G(ellin+ellout-p)
		numer.append(b)
		denom.append(a)

		numer.append((2*(ellin+ellout) - 4*p + 1))
		denom.append((2*(ellin+ellout) - 2*p + 1))

		q = ellin+ellout-2*p
		if front_coeff:
			numer.append((2*ellout+1))
			denom.append((2*q+1))

		numer = Fraction(scipy.prod(numer))
		denom = Fraction(scipy.prod(denom))
		if not as_string:
			coeffs.append(numer*1./denom)
			qvals.append(q)
		else:
			retstr.append('l{:d} {}'.format(q,numer/denom))

	if not as_string:
		return qvals[::-1], coeffs[::-1]
	else:
		return retstr[::-1]
	
class MultipoleToMultipole(object):
	
	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('MultiToMulti')

	def __init__(self,ellsin,ellsout,w):
		self.logger.info('Setting multipoles to multipoles transforms.')
		self.conversion = scipy.empty((len(ellsout),len(ellsin))+w(0,0).shape,dtype=w(0,0).dtype)
		for illout,ellout in enumerate(ellsout):
			for illin,ellin in enumerate(ellsin):
				ells,coeffs = coefficients(ellout,ellin[0]) #case ellin = (ell,n)
				self.conversion[illout][illin] = scipy.sum([coeff*w(ell,ellin[-1]) for ell,coeff in zip(ells,coeffs)],axis=0)
	
	def transform(self,func):
		#tmp = func*self.conversion
		return scipy.sum(func*self.conversion,axis=1)

		
	@utils.getstateclass
	def getstate(self,state):
		for key in ['conversion']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state
	
	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	@utils.loadstateclass
	def loadstate(self,state):
		self.setstate(state)
		return self	

class ConvolutionMultipole(object):
	
	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('Convolution')

	def __init__(self,**params):
		self.params = params

	@utils.classparams
	def set_grid(self,s,ellsin,ellsout):
		for key in ['s']: setattr(self,key,scipy.asarray(self.params[key],dtype=self.TYPE_FLOAT))
		self.multitomulti = MultipoleToMultipole(self.ellsin,self.ellsout,lambda ell,n: self.window[n](self.s,ell))
		#print self.multitomulti.conversion.shape
		#print self.multitomulti.conversion[...,self.s<5]
		#self.multitomulti.conversion[...,self.s<5] = 0

	@utils.classparams
	def set_window(self,path_window={}):
		self.window = {n: WindowFunction.load(path_window[n]) for n in path_window}
		self.ns = sorted(self.window.keys())
		assert self.ns[0] == 0
		self.los = self.window[0].los[0]
		for n in self.ns: assert self.window[n].los == (self.los,n)
		self.normwindow = self.window[0].norm
		return self.los, self.ns

	def convolve(self,func):
		return self.multitomulti.transform(func)

	@property
	def ellsin(self):
		return self.params['ellsin']

	@property
	def ellsout(self):
		return self.params['ellsout']

	def indexin(self,ell):
		return self.ellsin.index(ell)

	def indexout(self,ell):
		return self.ellsout.index(ell)

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','k']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['multitomulti']:
			if hasattr(self,key): state[key] = getattr(self,key).getstate()
		return state

	@utils.setstateclass
	def setstate(self,state):
		if 'multitomulti' in state: self.multitomulti = MultipoleToMultipole.loadstate(state['multitomulti'])

	@classmethod
	@utils.loadstateclass
	def loadstate(self,state):
		self.setstate(state)
		return self	

	def copy(self):
		return self.__class__.loadstate(self.getstate())
