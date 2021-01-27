import logging
import scipy
from pycute import PyCute
from . import utils

class BasePyCute(PyCute):

	logger = logging.getLogger('BasePyCute')

	def __init__(self,**params):
		super(BasePyCute,self).__init__()
		self.params = params

	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	def loadstate(cls,state):
		self = cls()
		self.setstate(state)
		return self

	def copy(self):
		new = self.__class__.loadstate(self.getstate())
		new.set_grid()
		return new

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()

	def __radd__(self,other):
		if other == 0: return self
		return self.__add__(other)

	def __iadd__(self,other):
		if other == 0: return self
		return self.__add__(other)

	def __rmul__(self,other):
		return self * other

	def __imul__(self,other):
		return self * other

	def normalize(self):
		self *= 1./self.weight_tot
		return self

class PyReal2PCF(BasePyCute):

	@utils.classparams
	def set_grid(self,sedges=[0.,120.],ssize=None,sbinning='lin',muedges=[-1.,1.],ells=[0,2,4,6,8,10,12],los='endpoint',losn=0,verbose='info'):

		self.set_verbosity(verbose)
		self.sedges = self.set_bin('main',edges=sedges,size=ssize,binning=sbinning)
		self.muedges = self.set_bin('aux',edges=muedges,size=1)
		self.ells = self.set_pole(ells=ells)
		self.los = self.set_los(1,los=los,n=losn)
	
	@utils.classparams
	def run(self,catalogue1,catalogue2,position='Position',weight='Weight',nthreads=8):
	
		pos = [catalogue1[position],catalogue2[position]]
		w = [catalogue1[weight],catalogue2[weight]]
		self.set_catalogues(pos,w)
		self.run_2pcf_multi(nthreads=nthreads)
		
		self.s = self.s.T[0]
		self.counts = self.counts.T
		
		self.weight = {key: self._weight[key].sum() for key in self._weight}
		self.weight_tot = scipy.prod([self.weight[key] for key in self.weight])
		
		return self

	@utils.getstateclass
	def getstate(self,state):
		for key in ['sedges','ells','los','s','counts','weight','weight_tot']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['catalogue1','catalogue2']: state['params'][key] = None
		return state
	
	def __add__(self,other):
		
		assert self.ells == other.ells
		if scipy.allclose(self.sedges,other.sedges,rtol=1e-04,atol=1e-05,equal_nan=False):
			self.s = (self.s*self.counts[0] + other.s*other.counts[0])/(self.counts[0] + other.counts[0])
			self.counts += other.counts
			self.weight_tot += other.weight_tot
		else:
			raise ValueError('s-edges are not compatible.')
		
		return self

	def __mul__(self,other):
		if isinstance(other,(scipy.floating,float)): 
			self.counts *= other
			self.weight_tot = self.weight_tot*other
		else:
			raise ValueError('Cannot multiply {} by {} object.'.format(self.__class__,type(other)))
		return self

	def rebin(self,factor=1):
		ns = len(self.s)//factor
		self.sedges = self.sedges[::factor]
		self.s = utils.bin_ndarray(self.s,(ns,),weights=self.counts[0],operation=scipy.sum)
		self.counts = utils.bin_ndarray(self.counts,(len(self.ells),ns),operation=scipy.sum)

class PyReal2PCFBinned(PyReal2PCF):

	@utils.classparams
	def set_grid(self,sedges=[0.,120.],ssize=None,sbinning='lin',binsize=1,muedges=[-1.,1.],ells=[0,2,4,6,8,10,12],los='endpoint',losn=0,verbose='info'):

		self.set_verbosity(verbose)
		self.sedges = self.set_bin('main',edges=sedges,size=ssize,binning=sbinning)
		self.muedges = self.set_bin('aux',edges=muedges,size=1)
		self.binedges = self.set_bin('bin',size=binsize)
		self.ells = self.set_pole(ells=ells)
		self.los = self.set_los(1,los=los,n=losn)
	
	@utils.classparams
	def run(self,catalogue1,catalogue2,position='Position',weight='Weight',bin='ibin',tobin=2,nthreads=8):
	
		pos = [catalogue1[position],catalogue2[position]]
		w = [catalogue1[weight],catalogue2[weight]]
		bins = [catalogue1[bin],catalogue2[bin]]
		self.set_catalogues(pos,w,bins)
		self.run_2pcf_multi_binned(tobin=tobin,nthreads=nthreads)
		
		self.counts = scipy.transpose(self.counts,axes=(2,0,1))  #counts is (ells,s,bin)
		
		self.weight = {key: self._weight[key].sum() for key in self._weight}
		self.weight_tot = 1
		for key in self.weight:
			if key == tobin: self.weight_tot *= scipy.bincount(self._bin[key].flatten().astype(int),weights=self._weight[key].flatten().astype(float),minlength=len(self.binedges)-1)
			else: self.weight_tot *= self._weight[key].sum()

		return self
	
	def __add__(self,other):
		
		assert self.ells == other.ells
		if scipy.allclose(self.sedges,other.sedges,rtol=1e-04,atol=1e-05,equal_nan=False):
			self.counts += other.counts
			self.weight_tot += other.weight_tot
		else:
			raise ValueError('s-edges are not compatible.')
		
		return self

	def __mul__(self,other):
		if isinstance(other,(int,float)): 
			self.counts *= other
			self.weight_tot = self.weight_tot*other
		elif isinstance(other,scipy.ndarray):
			self.counts = (self.counts*other).sum(axis=-1)
			self.weight_tot = (self.weight_tot*other).sum()
		else:
			raise ValueError('Cannot multiply {} by {} object.'.format(self.__class__,type(other)))
		return self

	def rebin(self,factor=1):
		ns = (len(self.sedges)-1)//factor
		self.sedges = self.sedges[::factor]
		self.counts = utils.bin_ndarray(self.counts,(len(self.ells),ns),operation=scipy.sum)

class PyRealKMu(BasePyCute):

	@utils.classparams
	def set_grid(self,muedges=[-1.,1.],mubinning='lin',musize=None,verbose='info'):

		self.set_verbosity(verbose)
		self.sedges = [0.,2.]
		self.set_bin('main',edges=self.sedges,binning='lin')
		self.muedges = self.set_bin('aux',edges=muedges,size=musize,binning=mubinning)
		
	@utils.classparams
	def run(self,catalogue1,catalogue2,position='Position',weight='Weight',nthreads=8):
	
		pos = [catalogue1[position],catalogue2[position]]
		w = [catalogue1[weight],catalogue2[weight]]
		self.set_catalogues(pos,w)
		self.run_2pcf_scos(nthreads=nthreads)

		for key in ['mu','counts']:
			setattr(self,key,scipy.squeeze(getattr(self,key)))

		self.weight = {key: self._weight[key].sum() for key in self._weight}

		return self

	@utils.getstateclass
	def getstate(self,state):
		for key in ['muedges','mu','counts','weight']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['catalogue1','catalogue2']: state['params'][key] = None
		return state

	def __add__(self,other):
		
		if scipy.allclose(self.muedges,other.muedges,rtol=1e-04,atol=1e-05,equal_nan=False):
			self.mu = (self.mu*self.counts + other.mu*other.counts)/(self.counts + other.counts)
			self.counts += other.counts
		else:
			raise ValueError('mu-edges are not compatible.')
		
		return self

