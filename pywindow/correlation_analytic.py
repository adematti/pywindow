import logging
import copy
import scipy
from scipy import special,constants,stats
from .pyreal2pcf import PyReal2PCF
from .pyreal3pcf import PyReal3PCF
from .pyanalytic import PyAnalytic2PCF,PyAnalytic3PCF,PyAnalytic4PCF
from .catalogue import Catalogue
from .correlation_catalogue import BaseCount,radial_volume,edges_to_mid
from .window_function import WindowFunction1D,WindowFunction2D
from . import utils

class BaseAngularCount(BaseCount):

	logger = logging.getLogger('BaseAngularCount')
	_NAME_TO_CLASS = {'PyReal2PCF':PyReal2PCF,'PyReal3PCF':PyReal3PCF}

	def normalize_catalogue(self,catalogues,ic='global',nside=32,seed=42,additional_bins=[]):
	
		assert ic in ['alpha','global','angular']
		return super(BaseAngularCount,self).normalize_catalogue(catalogues,ic=ic,nside=nside,seed=seed,additional_bins=additional_bins)

	@utils.classparams
	def prepare_catalogues(self,path_input,path_randoms,ic=None,nside=32,seed=42,ncuts=20,share='counts',subsample=2.,additional_bins=[]):
		catalogue = Catalogue.load(path_input)
		modes = path_randoms.keys()
		self.params['ic'] = self.ic
		catalogues = self.normalize_catalogue({mode: catalogue.deepcopy() for mode in modes},ic=self.ic,nside=nside,seed=seed,additional_bins=additional_bins)
		if isinstance(subsample,float): subsample = {mode: subsample for mode in modes} 
		for mode in modes: catalogues[mode].subsample(subsample[mode])
		if self.ic == 'angular': catalogues = self.slice_catalogues(catalogues,ncuts=ncuts,share=share)
		for mode,catalogue in catalogues.items(): catalogue.save(path_randoms[mode])


class Angular2PCF(BaseAngularCount):

	logger = logging.getLogger('Angular2PCF')
	
	@property
	def modes(self):
		if self.params.get('modes',None) is not None: return self.params['modes']
		if self.ic == 'global': return ['global','global']
		if self.ic == 'alpha': return ['alpha','alpha']

	@utils.classparams
	def set_angular(self,icut=0,ncuts=1,iaddbins=None,power_weight=1,modes=None):

		iaddbins = utils.tolist(iaddbins,n=2,value=None)
		power_weight = utils.tolist(power_weight,n=2,value=1)
		
		catalogues = []
		for imode,mode in enumerate(self.modes):
			path = self.params['path_randoms'][mode]
			catalogue = Catalogue.load(path)
			if iaddbins[imode] is not None:
				catalogue = catalogue[catalogue['iaddbin'] == iaddbins[imode]]
				self.logger.info('Additional cut {:d} for {} randoms ({:d} objects).'.format(iaddbins[imode],mode,catalogue.size))
			catalogue['Position'] /= catalogue.distance()[:,None]
			power = power_weight[imode]
			if power != 1:
				self.logger.info('Raising weights {} to power {:.4f}.'.format(mode,power))
				catalogue['Weight'] **= power
			catalogues.append(catalogue)

		catalogue1 = catalogues[0].slice(icut,ncuts)
		catalogue2 = catalogues[1]

		pyreal2pcf = PyReal2PCF(ells=[0],los='endpoint',losn=0,**self.params)
		pyreal2pcf.set_grid()
		self.result = pyreal2pcf.run(catalogue1,catalogue2)
		
	def to_window(self,**params):
	
		window = WindowFunction1D(**params)
		
		sedges = self.result.sedges
		window.poles = self.result.ells
		window.los = self.result.los
		if hasattr(self.result,'s'):
			window.s = self.result.s
			empty = scipy.isnan(window.s)
			window.s[empty] = edges_to_mid(sedges)[empty]
		else:
			window.s = edges_to_mid(sedges)

		window.window = self.result.counts
		volume = -scipy.diff(s_to_cos(sedges))
		window.window /= volume[None,...]
		
		window.pad_zero()
		
		window.s = s_to_cos(window.s)[::-1]
		window.window[0] = window.window[0][::-1]
		
		return window
	
class Angular3PCFBinned(BaseAngularCount):

	logger = logging.getLogger('Angular3PCFBinned')
	
	@property
	def modes(self):
		if self.params.get('modes',None) is not None: return self.params['modes']
		if self.ic == 'global': return ['alpha','alpha','global']
		if self.ic == 'radial': return ['alpha','alpha','radial']
		if self.ic == 'angular': return ['alpha','alpha','angular']

	@utils.classparams
	def set_angular(self,icut=0,icut1=0,ncuts=1,iaddbins=None,modes=None):

		iaddbins = utils.tolist(iaddbins,n=3,value=None)
		
		catalogues = []
		for imode,mode in enumerate(self.modes):
			path = self.params['path_randoms'][mode]
			catalogue = Catalogue.load(path)
			if iaddbins[imode] is not None:
				catalogue = catalogue[catalogue['iaddbin'] == iaddbins[imode]]
				self.logger.info('Additional cut {:d} for {} randoms ({:d} objects).'.format(iaddbins[imode],mode,catalogue.size))
			catalogue['Position'] /= catalogue.distance()[:,None]
			catalogues.append(catalogue)
		
		icutibins = catalogues[1].attrs['icutibins'][icut]
		assert (icutibins == catalogues[2].attrs['icutibins'][icut]).all()
		icutnbins = len(icutibins)
		
		catalogue1 = catalogues[0].slice(icut1,ncuts)
		
		pyreal3pcf = PyReal3PCF(ells=[0],los='endpoint',losn=0,**self.params)
		pyreal3pcf.set_grid()
		
		self.result = 0
		for ibin,bin in enumerate(icutibins):
			self.logger.info('Correlating slice {:d} ({:d}/{:d}).'.format(bin,ibin+1,icutnbins))
			catalogue2 = catalogues[1][catalogues[1]['ibin'] == bin]
			catalogue3 = catalogues[2][catalogues[2]['ibin'] == bin]
			if not catalogue2 or not catalogue3: continue
			self.result += pyreal3pcf.copy().run(catalogue1,catalogue2,catalogue3)

	def to_window(self,**params):
	
		window = WindowFunction2D(**params)
		
		ells = self.result.ells
		sedges = self.result.sedges
		counts = self.result.counts
		
		window.s = map(edges_to_mid,sedges)
		window.poles = [(ell1,ell2) for ell1 in ells[0] for ell2 in ells[1]]
		window.los = self.result.los
		window.window = counts.reshape((-1,)+counts.shape[2:]) #window.window is (ell,s,d)
		
		volume = scipy.prod(scipy.meshgrid(*[scipy.diff(s_to_cos(sedge)) for sedge in sedges],sparse=False,indexing='ij'),axis=0)
		window.window /= volume[None,...]

		window.pad_zero()

		window.s = [s_to_cos(s)[::-1] for s in window.s]
		window.window[0] = window.window[0][::-1,::-1]

		return window	

def s_to_cos(s):
	return 1.-s**2/2.

class BaseAnalyticCount(object):

	logger = logging.getLogger('BaseAnalyticCount')
	_NAME_TO_CLASS = {'PyAnalytic2PCF':PyAnalytic2PCF,'PyAnalytic3PCF':PyAnalytic3PCF,'PyAnalytic4PCF':PyAnalytic4PCF}
	
	def __init__(self,**params):
		self.params = params

	@utils.setstateclass
	def setstate(self,state):
		if 'result' in state:
			cls = self._NAME_TO_CLASS[state['result']['__class__']]
			self.result = cls.loadstate(state['result']['__dict__'])

	@utils.getstateclass
	def getstate(self,state):
		for key in ['result']:
			if hasattr(self,key):
				tmp = getattr(self,key)
				state[key] =  {'__class__':tmp.__class__.__name__, '__dict__':tmp.getstate()}
		return state

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self
		
	@utils.saveclass
	def save(self):
		return self.getstate()
	
	def set_angular(self,angular):
		self.angular = angular

	@utils.classparams
	def get_radial_density(self,catalogue,iaddbins=None,rwidth=2.,redges=None,normalize=True,density=True,power_weight=1):
		
		iaddbins = utils.tolist(iaddbins)
		normalize = utils.tolist(normalize,n=len(iaddbins),fill=-1)
		density = utils.tolist(density,n=len(iaddbins),fill=-1)
		power_weight = utils.tolist(power_weight,n=len(iaddbins),value=1)
		
		distance = catalogue.distance()
		dmin,dmax = distance.min(),distance.max()
		self.logger.info('Comoving distances: {:.1f} - {:.1f}.'.format(dmin,dmax))
			
		if redges is not None:
			radialedges = scipy.array(redges)
			rwidth = scipy.mean(scipy.diff(radialedges))
			rmin,rmax = radialedges.min(),radialedges.max()
			if (rmin>dmin) or (rmax<dmax): raise ValueError('Provided radial-edges ({:.1f} - {:.1f}) do not encompass the full survey ({:.1f} - {:.1f}).'.format(rmin,rmax,dmin,dmax))
			self.logger.info('Provided radial-edges of width: {:.1f} and range: {:.1f} - {:.1f}.'.format(rwidth,rmin,rmax))
			nbins = len(radialedges)-1
		else:
			self.logger.info('Provided radial-width: {:.1f}.'.format(rwidth))
			nbins = scipy.rint((dmax-dmin)/rwidth).astype(int)
			radialedges = scipy.linspace(dmin,dmax+1e-9,nbins+1)
			
		self.logger.info('There are {:d} radial-bins with an average of {:.1f} objects.'.format(nbins,len(catalogue)*1./nbins))
		
		def radial_density(distance,weight,normalize=True,density=True):
			toret = stats.binned_statistic(distance,values=weight,statistic='sum',bins=radialedges)[0]
			if density: toret /= radial_volume(radialedges)
			if normalize: toret /= toret.sum()
			return toret
		
		radial = (radialedges[:-1] + radialedges[1:])/2.
		
		densities,weights = [],[]
		for iaddbin_,normalize_,density_,power_ in zip(iaddbins,normalize,density,power_weight):
			if iaddbin_ is not None:
				mask = catalogue['iaddbin'] == iaddbin_
				self.logger.info('Additional cut {:d} ({:d} objects).'.format(iaddbin_,mask.sum()))
			else: mask = catalogue.trues()
			weight = catalogue['Weight'][mask]
			if power_ != 1:
				self.logger.info('Raising weights to power {:.4f}.'.format(power_))
				weight **= power_
			densities.append(radial_density(distance[mask],weight,normalize=normalize_,density=density_))
			weights.append(weight)
		
		self.params['catalogue'] = None

		return radial,densities,weights

	@utils.classparams
	def set_normalization(self,path_ref=None):
		if path_ref is None: path_ref = self.params['path_randoms']
		randoms = Catalogue.load(path_ref)
		self.normref = randoms.attrs['norm']
		self.weight_tot = self.result.integral()
		self.norm = self.normref*self.weight_tot

	@property
	def normalization(self):
		if not hasattr(self,'norm'): self.set_normalization()
		return self.norm

	def __add__(self,other):
		self.result += other.result
		return self

	def __radd__(self,other):
		if other == 0: return self
		return self.__add__(other)

	def __iadd__(self,other):
		if other == 0: return self
		return self.__add__(other)

class Analytic2PCF(BaseAnalyticCount):
	
	logger = logging.getLogger('Analytic2PCF')
	
	@utils.classparams
	def set_multipoles(self,s,iaddbins=None):
		
		catalogue = Catalogue.load(self.params['path_randoms'])
		radials,densities,weights = self.get_radial_density(catalogue,iaddbins=iaddbins,density=[False,True])
		
		pyanalytic2pcf = PyAnalytic2PCF(**self.params)
		self.result = pyanalytic2pcf.run(s,self.angular.s,self.angular.window[0],radials,densities,typewin='global')
		self.result.weight_tot = scipy.prod(map(scipy.sum,weights))
		self.result.rescale()

	def to_window(self,**params):
	
		window = WindowFunction1D(**params)
		
		window.poles = self.result.ells
		window.los = self.result.los
		
		window.s = self.result.x
		window.window = self.result.y
		
		for ill,ell in enumerate(window): window.window[ill] *= (2*ell+1) #legendre normalization
		
		window.window /= self.normalization
		
		window.norm = self.normref
		window.pad_zero()
		
		return window

class Analytic3PCF(BaseAnalyticCount):
	
	logger = logging.getLogger('Analytic3PCF')
	
	@utils.classparams
	def set_multipoles(self,s,ic='global',iaddbins=None):
		
		catalogue = Catalogue.load(self.params['path_randoms'])
		radials,densities,weights = self.get_radial_density(catalogue,iaddbins=iaddbins,density=[False,True,True])
		
		pyanalytic3pcf = PyAnalytic3PCF(**self.params)
		self.result = pyanalytic3pcf.run(s,self.angular.s,self.angular.window[0],radials,densities,typewin='{}-dlos'.format(ic))
		self.result.weight_tot = scipy.prod(map(scipy.sum,weights[:2]))
		self.result.rescale()
	
	def to_window(self,**params):
	
		window = WindowFunction2D(**params)
		
		ells = self.result.ells
		counts = self.result.y
		
		window.s = self.result.x
		window.poles = [(ell1,ell2) for ell1 in ells[0] for ell2 in ells[1]]
		window.los = self.result.los
		window.window = counts.reshape((-1,)+counts.shape[2:]) #window.window is (ell,s,d)
		
		for ill,(ell1,ell2) in enumerate(window): window.window[ill] *= (2*ell1+1)*(2*ell2+1) #legendre normalization
		
		window.window *= 2./self.normalization

		window.norm = self.normref		
		window.pad_zero()
		
		return window

class Analytic3PCFShotNoise(Analytic3PCF):
	
	logger = logging.getLogger('Analytic3PCFShotNoise')
	
	@utils.classparams
	def set_multipoles(self,s,ic='global',iaddbins=None):
		
		catalogue = Catalogue.load(self.params['path_randoms'])
		radials,densities,weights = self.get_radial_density(catalogue,iaddbins=[iaddbins]*2,density=[False,True],power_weight=[2,1])
		
		pyanalytic2pcf = PyAnalytic2PCF(**self.params)
		self.result = pyanalytic2pcf.run(s,self.angular.s,self.angular.window[0],radials,densities,typewin=ic)
		self.result.weight_tot = scipy.sum(weights[-1]**2)
		self.result.rescale()
	
	def to_window(self,**params):
	
		window = WindowFunction1D(**params)
		
		window.poles = self.result.ells
		window.los = self.result.los
		
		window.s = self.result.x
		window.window = self.result.y
		
		for ill,ell in enumerate(window): window.window[ill] *= (2*ell+1) #legendre normalization
		
		window.window *= 2./self.result.integral()
		
		window.norm = 1.
		window.pad_zero()
		
		return window

class Analytic4PCF(BaseAnalyticCount):
	
	logger = logging.getLogger('Analytic4PCF')
	
	@utils.classparams
	def set_multipoles(self,s,ic=['global,global'],iaddbins=None):
		
		catalogue = Catalogue.load(self.params['path_randoms'])
		radials,densities,weights = self.get_radial_density(catalogue,iaddbins=iaddbins,density=[False,True,False,True])
		
		pyanalytic4pcf = PyAnalytic4PCF(**self.params)
		pyanalytic4pcf.set_precision(calculation='costheta',n=10000,min=1.,max=-1.,integration='test')
		self.result = pyanalytic4pcf.run(s,self.angular.s,self.angular.window[0],radials,densities,typewin='-'.join(ic))
		self.result.weight_tot = scipy.prod(map(scipy.sum,weights[:2]))
		self.result.rescale()

	def to_window(self,**params):
	
		window = WindowFunction2D(**params)
		
		ells = self.result.ells
		counts = self.result.y
		
		window.s = self.result.x
		window.poles = [(ell1,ell2) for ell1 in ells[0] for ell2 in ells[1]]
		window.los = self.result.los
		window.window = counts.reshape((-1,)+counts.shape[2:]) #window.window is (ell,s,d)
		
		for ill,(ell1,ell2) in enumerate(window): window.window[ill] *= (2*ell1+1)*(2*ell2+1) #legendre normalization
		
		window.window /= self.normalization

		window.norm = self.normref		
		window.pad_zero()
		
		return window

class Analytic4PCFShotNoise(Analytic4PCF):
	
	logger = logging.getLogger('Analytic4PCFShotNoise')
	
	@utils.classparams
	def set_multipoles(self,s,ic=['global','global'],iaddbins=None):
		
		catalogue = Catalogue.load(self.params['path_randoms'])
		radials,densities,weights = self.get_radial_density(catalogue,iaddbins=[iaddbins]*2,density=[False,True],power_weight=[2 if 'radial' in ic else 1,1])
		
		pyanalytic2pcf = PyAnalytic2PCF(**self.params)
		if ('global' in ic) or (('radial' in ic) and ('angular' in ic)):
			typewin = 'global'
		else:
			typewin = ic[0]
		if typewin == 'angular':
			angular_s = None
			angular_window = None
		else:
			angular_s = self.angular.s
			angular_window = self.angular.window[0]
		self.result = pyanalytic2pcf.run(s,angular_s,angular_window,radials,densities,typewin=typewin)
		self.result.weight_tot = scipy.sum(weights[-1]**2)
		self.result.rescale()
	
	def to_window(self,**params):
	
		window = WindowFunction1D(**params)
		
		window.poles = self.result.ells
		window.los = self.result.los
		
		window.s = self.result.x
		window.window = self.result.y
		
		for ill,ell in enumerate(window): window.window[ill] *= (2*ell+1) #legendre normalization
		
		window.window *= 1./self.result.integral()
		
		window.norm = 1.
		window.pad_zero()
		
		return window
