import functools
import logging
import warnings
import numpy
import scipy
from scipy import constants,integrate,special,stats,interpolate
from .correlation_catalogue import BaseCount
from . import utils
from .pyreal2pcf import PyRealKMu
from .catalogue import Catalogue

class MuCount(BaseCount):
	
	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('MuCount')

	def __init__(self,**params):
		self.params = params
	
	@utils.classparams
	def save_ksampling(self,path_spectrum,path_ksampling=None,kmax=0.1):

		from analysis.utils import load_data

		self.logger.info('Loading power spectrum: {}.'.format(path_spectrum))
		get_data = load_data(path_spectrum,estimator='spectrum')
		self.kedges = get_data('edges')
		self.logger.info('Using k-edges: {:.4g} - {:.4g} ({:d}).'.format(self.kedges[0],self.kedges[-1],len(self.kedges)))
		self.kedges = self.kedges[self.kedges<kmax]
		self.logger.info('Using kmax = {:.4g}.'.format(kmax))

		axis = [0,1,2]
		Nmesh = get_data('Nmesh'); BoxSize = get_data('BoxSize'); kfun = 2.*constants.pi/BoxSize
		self.logger.info('Used grid {0[0]:d} x {0[1]:d} x {0[2]:d} with box size {1[0]:.4g} x {1[1]:.4g} x {1[2]:.4g} i.e. fundamental wavevector {2[0]:.4g} x {2[1]:.4g} x {2[2]:.4g}.'.format(Nmesh,BoxSize,kfun))
		halfNmesh = []
		for iaxis in axis:
			if Nmesh[iaxis] % 2 == 0: Nmesh[iaxis] += 1
			halfNmesh.append(Nmesh[0]//2)
		kcoords = [scipy.arange(Nmesh[0])-halfNmesh[0],scipy.arange(Nmesh[1])-halfNmesh[1],scipy.arange(halfNmesh[2])]
		for iaxis in axis: kcoords[iaxis] = kfun[iaxis]*kcoords[iaxis]
		kgrid = scipy.meshgrid(*kcoords,sparse=False,indexing='ij')
		for iaxis in axis:
			kgrid[iaxis][:halfNmesh[0],:,0] = scipy.inf
			kgrid[iaxis][:halfNmesh[0]+1,:halfNmesh[1]+1,0] = scipy.inf
			kgrid[iaxis][halfNmesh[0],halfNmesh[1],0] = 0.
			kgrid[iaxis] = kgrid[iaxis].flatten()
		kgrid = scipy.asarray(kgrid).T
		knorm = numpy.linalg.norm(kgrid,ord=2,axis=1)
		mask = knorm<self.kedges[-1]
		knorm = knorm[mask]
		kgrid = kgrid[mask]
		knorm[knorm==0.] = 1.
		kgrid /= knorm[:,None]
		self.k,_,kbinnumber = stats.binned_statistic(knorm,knorm,statistic='mean',bins=self.kedges)
		wgrid = scipy.ones((kgrid.shape[0]),dtype=kgrid.dtype)
		wgrid[knorm==0] = 0.5 #to avoid double counts of k=(0,0,0)
		self.logger.info('Keeping {:d} k-modes.'.format(len(wgrid)))

		catalogue = Catalogue({'Position':kgrid,'Weight':wgrid,'ikbin':kbinnumber-1},k=self.k,kedges=self.kedges,los=get_data('attrs').get('los',None))

		catalogue.save(path_ksampling)

	@utils.classparams
	def set_kmu(self,icut=0,ncuts=1,muedges=scipy.linspace(0.,1.,101),path_randoms=None,los=None):
		
		kcatalogue = Catalogue.load(self.params['path_ksampling'])
		self.k = kcatalogue.attrs['k']
		self.kedges = kcatalogue.attrs['kedges']
		los = kcatalogue.attrs.get('los',None)
		
		if los is not None:
			self.logger.info('Using los {0[0]:.2g} {0[1]:.2g} {0[2]:.2g}.'.format(los))
			randoms = Catalogue({'Position':[los],'Weight':[1.]})
		else:
			randoms = Catalogue.load(path_randoms).slice(icut,ncuts)

		randoms['Position'] = randoms['Position']/randoms.distance()[:,None]
	
		assert muedges[0] == 0 and (muedges>=0).all()
		muedges = scipy.concatenate([-muedges[-1:0:-1],muedges])
		pyreal2pcf = PyRealKMu(**self.params)
		pyreal2pcf.set_grid(muedges=muedges)
		self.result = []

		nkedges = len(self.kedges)-1
		for ikbin in range(nkedges):
			self.logger.info('Computing (k,mu) in ({:d}/{:d}) k-bin.'.format(ikbin+1,nkedges))
			mask = kcatalogue['ikbin'] == ikbin
			#print kcatalogue[mask]['Position'][:,2]
			self.result.append(pyreal2pcf.copy().run(randoms,kcatalogue[mask]))
			
	def __add__(self,other):
		for ires in range(len(self.result)):	
			self.result[ires] += other.result[ires]
		return self
	
	def to_window(self,**params):
	
		window = MuFunction(**params)
		
		window.k = self.k
		window.window = scipy.asarray([res.counts for res in self.result])

		with warnings.catch_warnings():
			warnings.simplefilter('ignore',category=RuntimeWarning)
			window.mu = scipy.ma.average([res.mu for res in self.result],weights=window.window,axis=0).data
		
		muedges = self.result[0].muedges
		mupositive = len(muedges)//2
		muedges = muedges[mupositive:]
		assert muedges[0] == 0 and (muedges >= 0).all()
		window.mu = scipy.mean([window.mu,-window.mu[::-1]],axis=0)[mupositive:] #-1 because we took half of the shell
		window.window = scipy.sum([window.window,window.window[:,::-1]],axis=0)[:,mupositive:]
		
		empty = scipy.isnan(window.mu)
		window.mu[empty] = edges_to_mid(muedges)[empty]
		window.error = scipy.sqrt(window.window)
		
		#print window.window.shape
		norm = scipy.sum(window.window*scipy.diff(muedges),axis=-1)/(muedges[-1]-muedges[0])
		window.window /= norm[:,None]
		window.error /= norm[:,None]

		window.pad_zero()
		
		return window

	@utils.getstateclass
	def getstate(self,state):
		for key in ['k','kedges']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['result']:
			if hasattr(self,key): state[key] = [el.getstate() for el in getattr(self,key)]
		return state

	@utils.setstateclass
	def setstate(self,state):
		for key in ['result']:
			if key in state: setattr(self,key,[PyRealKMu.loadstate(el) for el in state[key]])

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()


class MuFunction(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('MuFunction')

	def __init__(self,**params):
		self.params = params

	@utils.getstateclass
	def getstate(self,state):
		for key in ['k','mu','window','error']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()

	@classmethod
	def no_window(cls):
		self = cls()
		return self

	def __call__(self,k,mu,kind_interpol='linear'):
		shape = (len(k),len(mu))
		if not hasattr(self,'window'):
			return scipy.ones(shape,dtype=self.TYPE_FLOAT)
		interpolated_window = interpolate.interp2d(self.k,self.mu,self.window.T,kind=kind_interpol,bounds_error=False,fill_value=1.)
		return interpolated_window(k,mu).T
		
	def poisson_error(self,k,mu):
		interpolated_error = interpolate.interp2d(self.k,self.mu,self.error.T,kind=kind_interpol,bounds_error=False,fill_value=0.)
		return interpolated_error(k,mu).T
		
	def pad_zero(self):
		self.k = scipy.pad(self.k,pad_width=((1,0)),mode='constant',constant_values=0.)
		self.mu = scipy.pad(self.mu,pad_width=((1,1)),mode='constant',constant_values=((0,1)))
		self.window = scipy.pad(self.window,pad_width=((1,0),(1,1)),mode='edge')
		self.error = scipy.pad(self.error,pad_width=((1,0),(1,0)),mode='edge')

def edges_to_mid(edges):
	return (edges[1:]+edges[:-1])/2.
