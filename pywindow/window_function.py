import logging
import functools
import copy
import scipy
from scipy import interpolate
import re
from . import utils
	
def vectorize(func):
	@functools.wraps(func)				
	def wrapper(self,s,poles,*args,**kwargs):
	
		squeeze = False
		if scipy.isscalar(poles) or isinstance(poles,tuple):
			squeeze = True
			poles = [poles]
		if scipy.isscalar(s): squeeze = True
		res = scipy.asarray([func(self,s,pole,*args,**kwargs) for pole in poles])
		if squeeze:
			res = scipy.squeeze(res)
			if res.ndim == 0: return res[()]
		return res
	
	return wrapper

class BaseWindowFunction(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('BaseWindowFunction')

	def __init__(self,**params):
		self.params = params

	@classmethod
	def zeros(cls,zero=None):
		self = cls()
		self.poles = []
		if zero is not None:
			self.poles = [self.zero]
			self.s = scipy.asarray([0.,scipy.inf])
			if self.ndim>1: self.s = [self.s]*self.ndim
			self.window = scipy.asarray([zero]*2**self.ndim).reshape((1,)+(2,)*self.ndim)
		self.norm = 1.
		return self
	
	def index(self,pole):
		return self.poles.index(pole)
	
	def __contains__(self,pole):
		return pole in self.poles
		
	def __iter__(self):
		return self.poles.__iter__()

	@property
	def shape(self):
		return self.window.shape[1:]

	def pad_zero(self,nonmonopole_to_zero=True):
		if self.ndim == 1: self.s = scipy.pad(self.s,pad_width=((1,0)),mode='constant',constant_values=0.)
		else:
			for idim in range(self.ndim): self.s[idim] = scipy.pad(self.s[idim],pad_width=((1,0)),mode='constant',constant_values=0.)
		self.window = scipy.pad(self.window,pad_width=((0,0),)+((1,0),)*self.ndim,mode='edge')
		if hasattr(self,'error'): self.error = scipy.pad(self.error,pad_width=((1,0),)*self.ndim,mode='edge')
		if nonmonopole_to_zero:
			if self.ndim == 1:
				for pole in self:
					if pole != self.zero: self.window[self.index(pole),0] = 0.
			else:
				for pole in self:
					for idim in range(self.ndim):
						if pole[idim] != 0: utils.fill_axis(self.window,axis=(0,1+idim),slices=(self.index(pole),0),values=0.)

	def __or__(self,other):
		priority_normalize = 'first'
		import copy
		new = self.deepcopy()
		new.params.update(copy.deepcopy(other.params))
		for key in ['los','ndim']: assert getattr(self,key) == getattr(other,key)
		if scipy.amax(self.s) <= scipy.amax(other.s):
			first = self
			second = other
		else:
			first = other
			second = self
		#assert (first.index(first.zero) == 0) and (second.index(second.zero) == 0)
		if self.ndim == 1:
			firsts = [first.s]
			seconds = [second.s]
		else:
			firsts = first.s
			seconds = second.s
		firstshape = scipy.asarray(first.window.shape[1:])
		firstipoles = [ipole for ipole,pole in enumerate(first.poles) if pole in second.poles]
		secondipoles = [second.poles.index(first.poles[ipole]) for ipole in firstipoles]
		new.poles = [first.poles[ipole] for ipole in firstipoles]
		secondmask = [s2 >= s1[-1] for s1,s2 in zip(firsts,seconds)]
		new.s = [scipy.concatenate([s1,s2[mask2]],axis=-1) for s1,s2,mask2 in zip(firsts,seconds,secondmask)]
		if self.ndim == 1: new.s = new.s[0]
		overlaps = [s2[~mask2] for s2,mask2 in zip(seconds,secondmask)]
		ratio = first(overlaps,first.zero,kind_interpol='linear')/second(overlaps,second.zero,kind_interpol='linear')
		#norm = scipy.mean(ratio)
		slices = (slice(1,None),)*self.ndim
		norm = scipy.mean(ratio[slices]) #do not take (imprecise or padded) first point
	
		def normalize_first_second(first,second,norm,priority_normalize):
			firstnorm = secondnorm = 1.
			if priority_normalize == 'second': norm = 1./norm
			if getattr(first,'norm',None) is not None: new.norm = first.norm			
			if getattr(second,'norm',None) is None:
				secondnorm = norm
				self.logger.info('Rescaling {} part of the window function by {:.3f}.'.format(priority_normalize,secondnorm))
			elif getattr(first,'norm',None) is None:
				firstnorm = 1/norm
				new.norm = second.norm
				self.logger.info('Rescaling {} part of the window function by {:.3f}.'.format('second' if priority_normalize=='first' else 'first',firstnorm))
			else:
				self.logger.info('No rescaling, as both window functions are normalized (first over second ratio found: {:.3f}).'.format(norm))
			return firstnorm,secondnorm
		
		if priority_normalize == 'first':
			firstnorm,secondnorm = normalize_first_second(first,second,norm,priority_normalize)
		else:
			secondnorm,firstnorm = normalize_first_second(second,first,norm,priority_normalize)

		new.window = secondnorm*second(new.s,new.poles,kind_interpol='linear')
		slices = (slice(None),) + tuple(slice(0,end) for end in firstshape)
		new.window[slices] = firstnorm*first.window[...]
		if hasattr(second,'error'):
			new.error = secondnorm*second.poisson_error(new.s,kind_interpol='linear')
			if hasattr(first,'error'): new.error[slices[1:]] = firstnorm*first.error[...]
		return new

	@utils.setstateclass
	def setstate(self,state):
		pass

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','window','error','poles','los','norm','ndim']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

	@classmethod
	@utils.loadstateclass
	def loadstate(self,state):
		self.setstate(state)
		return self

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self
		
	@utils.saveclass
	def save(self):
		return self.getstate()

	def copy(self):
		return self.__class__.loadstate(self.getstate())
	
	def deepcopy(self):
		import copy
		return copy.deepcopy(self)

	def __ror__(self,other):
		if other == 0: return self
		return self.__or__(other)

	def __ior__(self,other):
		if other == 0: return self
		return self.__or__(other)

	def __add__(self,other):
		import copy
		new = self.deepcopy()
		new.params.update(copy.deepcopy(other.params))
		#for key in ['los','ndim']: assert getattr(self,key) == getattr(other,key)
		new.window += other(new.s,new.poles)
		return new

	def __radd__(self,other):
		if other == 0: return self
		return self.__add__(other)

	def __iadd__(self,other):
		if other == 0: return self
		return self.__add__(other)
		
	def __neg__(self):
		new = self.deepcopy()
		new.window *= -1
		return new
	
	def __sub__(self,other):
		return self.__add__(other.__neg__())

	def __rsub__(self,other):
		if other == 0: return self.__neg__()
		return self.__sub__(other).__neg__()

	def __isub__(self,other):
		if other == 0: return self
		return self.__sub__(other)

class WindowFunction1D(BaseWindowFunction):

	TYPE_FLOAT = scipy.float64
	zero = 0
	ndim = 1
	logger = logging.getLogger('WindowFunction1D')
	
	@vectorize
	def __call__(self,s,pole,kind_interpol='linear'):
		if scipy.isscalar(s): s = [s]
		shape = (len(s))
		if not hasattr(self,'window'):
			return scipy.zeros(shape,dtype=self.TYPE_FLOAT)
		if pole in self:
			interpolated_window = interpolate.interp1d(self.s,self.window[self.index(pole)],kind=kind_interpol,bounds_error=False,fill_value=0.)
			return interpolated_window(s)
		self.logger.warning('Sorry, pole {} is not calculated.'.format(pole))
		return scipy.zeros(shape,dtype=self.TYPE_FLOAT)
		
	def poisson_error(self,s,kind_interpol='linear'):
		interpolated_error = interpolate.interp1d(self.s,self.error,kind=kind_interpol,bounds_error=False,fill_value=0.)
		return interpolated_error(s)
	
	@property
	def ells(self):
		return self.poles

class WindowFunction2D(BaseWindowFunction):

	TYPE_FLOAT = scipy.float64
	zero = (0,0)
	ndim = 2
	logger = logging.getLogger('WindowFunction2D')

	def __init__(self,**params):
		self.params = params
			
	@property
	def ells(self):
		return [scipy.unique([pole[i] for pole in self.poles]).tolist() for i in [0,1]]
	
	@vectorize
	def __call__(self,s,pole,kind_interpol='linear'):
		ell1,ell2 = pole
		s = [[s_] if scipy.isscalar(s_) else s_ for s_ in s]
		shape = map(len,s)
		if not hasattr(self,'window'):
			return scipy.zeros(shape,dtype=self.TYPE_FLOAT)
		if pole in self:
			interpolated_window = interpolate.interp2d(self.s[0],self.s[-1],self.window[self.index(pole)].T,kind=kind_interpol,bounds_error=False,fill_value=0.)
			return interpolated_window(*s).T
		self.logger.warning('Sorry, pole {} is not calculated.'.format(pole))
		return scipy.zeros(shape,dtype=self.TYPE_FLOAT)
		
	def poisson_error(self,s,kind_interpol='linear'):
		interpolated_error = interpolate.interp2d(self.s[0],self.s[-1],self.error.T,kind=kind_interpol,bounds_error=False,fill_value=0.)
		return interpolated_error(*s).T

class WindowFunction(BaseWindowFunction):

	logger = logging.getLogger('WindowFunction')
	
	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		#self.ndim = self.window.ndim - 1
		#self.los = ('endpoint',0)
		if self.ndim == 1:
			new = object.__new__(WindowFunction1D)
			new.__dict__ = self.__dict__
		if self.ndim == 2:
			new = object.__new__(WindowFunction2D)
			new.__dict__ = self.__dict__
		return new

class TemplateSystematics(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('TemplateSystematics')

	def __init__(self,**params):
		self.params = params

	def index(self,pole):
		return self.poles.index(pole)
	
	def __contains__(self,pole):
		return pole in self.poles
		
	def __iter__(self):
		return self.poles.__iter__()
	
	def ellsin(self,ellout):
		toret = []
		for pole in self.poles:
			if pole[0] == ellout: toret.append(pole[1])
		return toret
		
	@property
	def ellsout(self):
		return self.params['ellsout']

	@property
	def kout(self):
		return self.params['kout']

	@utils.classparams
	def setup(self,kout,ellsout):
		self.kernel = {}
		for illout,ellout in enumerate(self.ellsout):
			ellsin = self.ellsin(ellout) 
			for ellin in ellsin: self.kernel[(ellout,ellin)] = self(self.kout[illout],(ellout,ellin))

	def model(self,**kwargs):
		toret = {ellout:0. for ellout in self.ellsout}
		for par in kwargs:
			if re.match('c[1-9]',par):
				p = int(par.replace('c',''))
				for ellout in self.ellsout:
					toret[ellout] += kwargs[par]*self.kernel[(ellout,p)]
		return [toret[ellout] for ellout in self.ellsout]
		
	def gradient(self,parameters):
		toret = {}
		for par in parameters:
			toret[par] = [0. for ellout in self.ellsout]
			if re.match('c[1-9]',par):
				p = int(par.replace('c',''))
				for illout,ellout in enumerate(self.ellsout): toret[par][illout] += self.kernel[(ellout,p)]
		return toret

	@vectorize
	def __call__(self,k,pole,kind_interpol='linear'):
		if scipy.isscalar(k): k = [k]
		shape = (len(k))
		if not hasattr(self,'window'):
			return scipy.zeros(shape,dtype=self.TYPE_FLOAT)
		if pole in self:
			interpolated_window = interpolate.interp1d(self.k,self.window[self.index(pole)],kind=kind_interpol,bounds_error=False,fill_value=0.)
			return interpolated_window(k)
		self.logger.warning('Sorry, pole {} is not calculated.'.format(pole))
		return scipy.zeros(shape,dtype=self.TYPE_FLOAT)

	@utils.getstateclass
	def getstate(self,state):
		for key in ['k','poles','window','kernel']:
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

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()
