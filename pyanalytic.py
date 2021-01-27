import scipy
from scipy import constants
from pyanacorr import PyAnacorr
from . import utils

def weights_trapz(x):
	trapzw = 4.*constants.pi*x**2*scipy.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.
	return trapzw

def integrate_trapz_1d(func,x):
	return scipy.sum(func*weights_trapz(x))

def integrate_trapz_2d(func,x):
	weights = weights_trapz(x[0])[:,None]*weights_trapz(x[1])
	return scipy.sum(func*weights)

class BasePyAnacorr(PyAnacorr):

	def __init__(self,**params):
		super(BasePyAnacorr,self).__init__()
		self.params = params

	@utils.setstateclass
	def setstate(self,state):
		pass

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','ells','los','x','y','weight_tot']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['costheta','angular','distance','radial']: state['params'][key] = None
		return state

	@classmethod
	def loadstate(cls,state):
		self = cls()
		self.setstate(state)
		return self

	def copy(self):
		new = self.__class__.loadstate(self.getstate())
		return new

	def __radd__(self,other):
		if other == 0: return self
		return self.__add__(other)

	def __iadd__(self,other):
		if other == 0: return self
		return self.__add__(other)

	def __add__(self,other):
		self.y += other.y
		if hasattr(self,'weight_tot') and hasattr(other,'weight_tot'):
			self.weight_tot += other.weight_tot
		return self
	
	def rescale(self):
		self.y *= self.weight_tot/self.integral()

class PyAnalytic2PCF(BasePyAnacorr):

	@utils.classparams
	def run(self,s,costheta,angular,distance,radial,ells=[0,2,4,6,8,10,12],los='midpoint',losn=0,typewin='global',nthreads=8):
		
		#self.set_precision(calculation='costheta',n=2000,min=1.,max=-1.,integration='test')
		self.set_2pcf_multi(s,costheta,angular,distance,radial,ells=ells,los=los,losn=losn,typewin=typewin,nthreads=nthreads)
		self.x = self.x[0]
		self.y = self.y.T
		
		return self
	
	def index(self,ell):
		return self.ells.index(ell)

	def integral(self):
		return integrate_trapz_1d(self.y[self.index(0)],self.x)

class PyAnalytic3PCF(BasePyAnacorr):

	@utils.classparams
	def run(self,s,costheta,angular,distance,radial,ells=[0,2,4,6,8,10,12],los='midpoint',losn=0,typewin='global',nthreads=8):
		
		s = utils.tolist(s,n=2,fill=0)
		#self.set_precision(calculation='costheta',n=10000,min=1.,max=-1.,integration='test')
		s[-1] = s[0] + 0.5*(s[0][1]-s[0][0])
		self.set_3pcf_multi(s,costheta,angular,distance,radial,ells=ells,los=los,losn=losn,typewin=typewin,nthreads=nthreads)
		self.y = scipy.transpose(self.y,axes=(2,3,0,1))
		
		return self

	def indexout(self,ell):
		return self.ells[0].index(ell)
		
	def indexin(self,ell):
		return self.ells[1].index(ell)

	def integral(self):
		return integrate_trapz_2d(self.y[self.indexout(0),self.indexin(0)],self.x)

class PyAnalytic4PCF(BasePyAnacorr):

	@utils.classparams
	def run(self,s,costheta,angular,distance,radial,ells=[0,2,4,6,8,10,12],los='midpoint',losn=0,typewin='global',nthreads=8):
		
		s = utils.tolist(s,n=2,fill=0)
		self.set_4pcf_multi(s,costheta,angular,distance,radial,ells=ells,los=los,losn=losn,typewin=typewin,nthreads=nthreads)
		self.y = scipy.transpose(self.y,axes=(2,3,0,1))
		
		return self

	def indexout(self,ell):
		return self.ells[0].index(ell)
		
	def indexin(self,ell):
		return self.ells[1].index(ell)

	def integral(self):
		return integrate_trapz_2d(self.y[self.indexout(0),self.indexin(0)],self.x)

