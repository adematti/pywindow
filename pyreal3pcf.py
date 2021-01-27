import scipy
from .pyreal2pcf import BasePyCute
from . import utils

class PyReal3PCF(BasePyCute):

	@utils.classparams
	def set_grid(self,sedges=[0.,120.],ssize=None,sbinning='lin',muedges=[-1.,1.],ells=[0,2,4,6,8,10,12],los='endpoint',losn=0,verbose='info'):

		self.set_verbosity(verbose)
		self.sedges = [self.set_bin('main',edges=sedges,size=ssize,binning=sbinning)]*2
		self.muedges = [self.set_bin('aux',edges=muedges,size=1)]*2
		self.ells = self.set_n_poles(ells,n=2)
		self.los = self.set_n_los(los,losn,n=2)
	
	@utils.classparams
	def run(self,catalogue1,catalogue2,catalogue3=None,position='Position',weight='Weight',nthreads=8):
	
		pos = [catalogue1[position],catalogue2[position]]
		w = [catalogue1[weight],catalogue2[weight]]
		if (self.ells[-1] != self.ells[0]) and (catalogue3 is None): catalogue3 = catalogue2
		if catalogue3 is not None:
			pos.append(catalogue3[position])
			w.append(catalogue3[weight])
		self.set_catalogues(pos,w)
		self.run_3pcf_multi_double_los(nthreads=nthreads)
		
		self.counts = scipy.transpose(self.counts,axes=(2,3,0,1))  #counts is (ells,elld,s,d)
		
		self.weight = {key: self._weight[key].sum() for key in self._weight}
		self.weight_tot = scipy.prod([self.weight[key] for key in self.weight])
		
		return self

	@utils.getstateclass
	def getstate(self,state):
		for key in ['sedges','ells','los','counts','weight','weight_tot']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['catalogue1','catalogue2','catalogue3']: state['params'][key] = None
		return state
	
	def __add__(self,other):

		assert self.ells == other.ells
		if scipy.allclose(self.sedges,other.sedges,rtol=1e-04,atol=1e-05,equal_nan=False):
			self.counts += other.counts
			self.weight_tot += other.weight_tot
		else:
			raise ValueError('s-edges are not compatible.')
		
		return self

	def rebin(self,factor=1):
		ns = (len(self.sedges[0])-1)//factor
		self.sedges = [sedges[::factor] for sedges in self.sedges]
		shape = tuple(len(ells) for ells in self.ells) + (ns,)*2
		self.counts = utils.bin_ndarray(self.counts,shape,operation=scipy.sum)

