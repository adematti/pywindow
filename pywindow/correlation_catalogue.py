import logging
import copy
import scipy
from scipy import special,constants
from .pyreal2pcf import PyReal2PCF,PyReal2PCFBinned
from .pyFFT2PCF import PyFFT2PCF,get_mesh,define_cartesian_box
from .pyreal3pcf import PyReal3PCF
from .pyreal4pcf import PyReal4PCFBinned
from .window_function import WindowFunction1D,WindowFunction2D
from .catalogue import Catalogue
from . import utils

class BaseCount(object):

	logger = logging.getLogger('BaseCount')
	_NAME_TO_CLASS = {'PyReal2PCF':PyReal2PCF,'PyReal2PCFBinned':PyReal2PCFBinned,'PyFFT2PCF':PyFFT2PCF,'PyReal3PCF':PyReal3PCF,'PyReal4PCFBinned':PyReal4PCFBinned}
	
	def __init__(self,**params):
		self.params = params

	@utils.setstateclass
	def setstate(self,state):
		if getattr(self,'result',0):
			cls = self._NAME_TO_CLASS[state['result']['__class__']]
			self.result = cls.loadstate(state['result']['__dict__'])

	@utils.getstateclass
	def getstate(self,state):
		for key in ['result']:
			tmp = getattr(self,key,0)
			if tmp: state[key] = {'__class__':tmp.__class__.__name__, '__dict__':tmp.getstate()}
			else: state[key] = tmp
		return state

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self
		
	@utils.saveclass
	def save(self):
		return self.getstate()
	
	def normalize_catalogue(self,catalogues,ic='global',rwidth=2.,redges=None,nside=32,seed=42,additional_bins=[]):
	
		assert ic in ['alpha','global','radial','angular']
		
		if not additional_bins: additional_bins = [lambda catalogue: catalogue.trues()]
		naddbins = len(additional_bins)
		self.logger.info('I include {:d} additional bins.'.format(naddbins))
		
		for mode,catalogue in catalogues.items():
			catalogue['iaddbin'] = catalogue.zeros(dtype=scipy.int8)
			catalogue.attrs['naddbins'] = naddbins
			for iaddbin,addbin in enumerate(additional_bins):
				mask = addbin(catalogue)
				catalogue['iaddbin'][mask] = iaddbin
				masksum = mask.sum()
				self.logger.info('Additional cut {:d} represents {:.1f}% ({:d}/{:d}) of {} randoms.'.format(iaddbin+1,masksum*100./len(mask),masksum,len(mask),mode))
				
		def weight_global(catalogue):
			
			self.logger.info('Global integral constraint.')

			for iaddbin in range(catalogue.attrs['naddbins']):
				mask = catalogue['iaddbin'] == iaddbin
				catalogue['Weight'][mask] /= catalogue['Weight'][mask].sum()
			
			attrs = {'nbins':1,'icutibins':[scipy.array([0],dtype=int)]}

			def bin(catalogue):
				return scipy.zeros((len(catalogue)),dtype=scipy.int8)
			
			return attrs,bin

		def weight_radial(catalogue,rwidth=rwidth,redges=redges):
		
			self.logger.info('Radial integral constraint.')
		
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
			ibin = scipy.digitize(distance,radialedges,right=False)-1
			
			for iaddbin in range(catalogue.attrs['naddbins']):
				mask = catalogue['iaddbin'] == iaddbin
				wcounts = scipy.bincount(ibin[mask],weights=catalogue['Weight'][mask],minlength=nbins)
				catalogue['Weight'][mask] /= wcounts[ibin[mask]]

			attrs = {'radialedges':radialedges,'nbins':nbins}

			def bin(catalogue):
				return scipy.digitize(catalogue.distance(),radialedges,right=False)-1
				
			return attrs,bin
				
		def weight_angular(catalogue,nside=nside):
			
			self.logger.info('Angular integral constraint.')
			
			import healpy
			pixarea = healpy.nside2pixarea(nside,degrees=True)
			npix = healpy.nside2npix(nside)
			self.logger.info('Pixels with nside = {:d}: {:.1f} square degree ({:d}).'.format(nside,pixarea,npix))
			
			#weights
			theta,phi = healpy.vec2ang(catalogue['Position'])
			ra,dec = phi/constants.degree,90.-theta/constants.degree
			self.logger.info('RA x DEC: [{:.1f}, {:.1f}] x [{:.1f}, {:.1f}].'.format(ra.min(),ra.max(),dec.min(),dec.max()))
			pix = healpy.ang2pix(nside,theta,phi,nest=False)
			counts = scipy.bincount(pix,minlength=npix)
			mask = counts > 0
			nbins = mask.sum()
			self.logger.info('There are {:d} pixels with an average of {:.1f} objects.'.format(nbins,len(catalogue)*1./nbins))
			pixtoibin = -scipy.ones((npix),dtype=scipy.int64)
			pixtoibin[mask] = scipy.arange(nbins)

			for iaddbin in range(catalogue.attrs['naddbins']):
				mask = catalogue['iaddbin'] == iaddbin
				wcounts = scipy.bincount(pix[mask],weights=catalogue['Weight'][mask])
				catalogue['Weight'][mask] /= wcounts[pix[mask]]
			
			attrs = {'nside':nside,'nbins':nbins}
			
			def bin(catalogue):
				theta,phi = healpy.vec2ang(catalogue['Position'])
				pix = healpy.ang2pix(nside,theta,phi,nest=False)
				return pixtoibin[pix]
		
			return attrs,bin
		
		def bin_catalogues(catalogues,attrs,bin):
			
			for mode,catalogue in catalogues.items():

				catalogue.attrs.update(attrs)
				catalogue['ibin'] = bin(catalogue)
				ibinmin,ibinmax = catalogue['ibin'].min(),catalogue['ibin'].max()
				self.logger.info('ibin range in {} randoms: {:d} - {:d}.'.format(mode,ibinmin,ibinmax))
				assert (ibinmin>=0) and (ibinmax<catalogue.attrs['nbins'])
		
		catalogue = catalogues[ic]
		
		if ic == 'global':
			
			attrs,bin = weight_global(catalogue)
			bin_catalogues(catalogues,attrs,bin)
		
		if ic == 'radial':
		
			attrs,bin = weight_radial(catalogue)
			bin_catalogues(catalogues,attrs,bin)
		
		if ic == 'angular':
			
			attrs,bin = weight_angular(catalogue)
			bin_catalogues(catalogues,attrs,bin)

		self.logger.info('Sum of weights: {:.1f}.'.format(catalogue['Weight'].sum()))

		return catalogues


	def slice_catalogues(self,catalogues,ncuts=20,share='counts'):
	
		self.logger.info('Cutting catalogues in {:d} slices for parallelization.'.format(ncuts))
		
		def share_slices(counts):
			cumcounts = scipy.cumsum(counts)
			cedges = scipy.linspace(0,cumcounts[-1]+1,ncuts+1)
			cutnumber = scipy.digitize(cumcounts,cedges)-1
			assert (cutnumber>=0).all() and (cutnumber<ncuts).all()
			return [scipy.flatnonzero(cutnumber == icut) for icut in range(ncuts)]
	
		modes = list(catalogues.keys())
		nbins = catalogues[modes[0]].attrs['nbins']
		for _,catalogue in catalogues.items(): assert catalogue.attrs['nbins'] == nbins

		ibin = scipy.concatenate([catalogue['ibin'] for _,catalogue in catalogues.items()])
		counts = scipy.bincount(ibin,minlength=nbins)
		
		if share == 'counts':
			self.logger.info('Count share.')
			sharedslices = share_slices(counts)
		else:
			self.logger.info('Flat share.')
			sharedslices = share_slices(scipy.ones_like(counts))
		assert sum(map(len,sharedslices)) == nbins
		
		for _,catalogue in catalogues.items(): catalogue.attrs['icutibins'] = sharedslices

		return catalogues

	@utils.classparams
	def prepare_catalogues(self,path_input,path_randoms,ic=None,rwidth=2.,redges=None,nside=32,seed=42,ncuts=20,share='counts',downsample=2.,additional_bins=[]):
		catalogue = Catalogue.load(path_input)
		modes = path_randoms.keys()
		self.params['ic'] = self.ic
		catalogues = self.normalize_catalogue({mode: catalogue.deepcopy() for mode in modes},ic=self.ic,rwidth=rwidth,redges=redges,nside=nside,seed=seed,additional_bins=additional_bins)
		if isinstance(downsample,float): downsample = {mode: downsample for mode in modes} 
		for mode in modes: catalogues[mode] = catalogues[mode].downsample(downsample[mode])
		if self.ic in ['angular','radial']: catalogues = self.slice_catalogues(catalogues,ncuts=ncuts,share=share)
		for mode,catalogue in catalogues.items(): catalogue.save(path_randoms[mode])
		
	@utils.classparams
	def set_normalization(self,path_ref=None):
		if path_ref is None: path_ref = list(self.params['path_randoms'].values())[0]
		randoms = Catalogue.load(path_ref)
		self.normref = randoms.attrs['norm']
		self.norm = self.normref*self.weight_tot
		self.shotnoise = (randoms['Weight']**2).sum()/self.norm
	
	@property
	def ic(self):
		ic = self.params.get('ic',None)
		if ic is None:
			ic = 'alpha'
			for mode in self.params['path_randoms']:
				if mode in ['global','radial','angular']:
					ic = mode
					break
		return ic

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
	
	@property	
	def weight_tot(self):
		return self.result.weight_tot

	def rebin(self,factor=1):
		self.result.rebin(factor=factor)

	def normalize(self):
		self.result.normalize()
		return self
	
class Real2PCF(BaseCount):

	logger = logging.getLogger('Real2PCF')
	
	@property
	def modes(self):
		if self.params.get('modes',None) is not None: return self.params['modes']
		if self.ic == 'global': return ['global','global']
		if self.ic == 'alpha': return ['alpha','alpha']

	@utils.classparams
	def set_multipoles(self,icut=0,ncuts=1,losn=0,iaddbins=[None,None],modes=None):
		
		catalogues = []
		for imode,mode in enumerate(self.modes):
			path = self.params['path_randoms'][mode]
			catalogue = Catalogue.load(path)
			if iaddbins[imode] is not None:
				catalogue = catalogue[catalogue['iaddbin'] == iaddbins[imode]]
				self.logger.info('Additional cut {:d} for {} randoms ({:d} objects).'.format(iaddbins[imode],mode,catalogue.size))
			catalogues.append(catalogue)

		catalogues[0] = catalogues[0].slice(icut,ncuts)

		pyreal2pcf = PyReal2PCF(**self.params)
		pyreal2pcf.set_grid()
		self.result = pyreal2pcf.run(catalogues[0],catalogues[1])
		
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
		if window.zero in window: window.error = window.window[window.index(window.zero)]**(1./2.)

		volume = (4.*constants.pi)*radial_volume(sedges)
		for ill,ell in enumerate(window): window.window[ill] *= (2*ell+1)/volume #legendre normalization and radial volume

		window.window /= self.normalization
		if hasattr(window,'error'): window.error /= volume*self.normalization
		window.norm = self.normref
		window.pad_zero()
		
		return window
	
	def to_4pcf(self,other,normalize=False):

		new = Real4PCFBinned(**self.params)
		new.params['path_randoms'].update(other.params['path_randoms'])
		if new.params.get('modes',None) is not None and other.params('modes',None) is not None: new.params['modes'].append(other.params['modes'])
		result = object.__new__(PyReal4PCFBinned)
		result.params = {}
		for inst in [self,other]: result.params.update(getattr(inst,'result').params)
		for key in ['sedges','ells','los']: setattr(result,key,[getattr(self.result,key),getattr(other.result,key)])
		#result.counts = scipy.einsum('ij,kl->ikjl',self.result.counts,other.result.counts)
		result.counts = scipy.transpose(self.result.counts[...,None,None]*other.result.counts,axes=(0,2,1,3))
		result.weight_tot = self.weight_tot*other.weight_tot
		result.weight = {key:item for key,item in self.result.weight.items()}
		result.weight.update({key+2:item for key,item in other.result.weight.items()})
		new.result = result
		
		return new

	def to_4pcfsn(self,iaddbins=None,path_ref=None):
		
		new = Real4PCFBinnedShotNoise(**self.params)

		if path_ref is None: path_ref = list(self.params['path_randoms'].values())[0]
		randoms = Catalogue.load(path_ref)
		if iaddbins is not None:
			weights = randoms['Weight'][randoms['iaddbin'] == iaddbins[0]]
		else:
			weights = randoms['Weight']
		weight = (weights**2).sum()/weights.sum()**2

		new.result = self.result*weight
		
		return new


class FFT2PCF(Real2PCF):

	logger = logging.getLogger('FFT2PCF')
	
	@utils.classparams
	def set_multipoles(self,losn=0,modes=None,iaddbins=[None,None]):
		
		meshs = []
		weight_tot = 1
		for imode,mode in enumerate(self.modes):
			path = self.params['path_randoms'][mode]
			catalogue = Catalogue.load(path)
			if iaddbins[imode] is not None: catalogue = catalogue[catalogue['iaddbin'] == iaddbins[imode]]
			weight_tot *= catalogue['Weight'].sum()
			if imode == 0: catalogue['Weight'] /= catalogue.distance()**losn
			meshs.append(get_mesh(catalogue.to_nbodykit(),position='Position',weight='Weight',mesh=self.params['mesh']))
		if len(meshs)>1:
			second = meshs[1]
		else:
			weight_tot **= 2
			second = None
		self.result = PyFFT2PCF(meshs[0],poles=self.params['ells'],second=second,kmin=0.)
		self.result.weight_tot = weight_tot

	def to_window(self,s,key='A',**params):
	
		poles = self.result.poles
		attrs = self.result.attrs
		self.logger.info('k-range: {:.1f} - {:.1f}.'.format(attrs['kfun'].max(),attrs['knyq'].min()))
	
		window = WindowFunction1D(**params)
		window.poles = attrs['poles']
		window.los = ('endpoint',self.params['losn'])
		window.s = scipy.asarray(s)
		
		self.logger.info('Normalization is {:.7g} and shotnoise is {:.7g}.'.format(self.normalization,self.shotnoise))
		#mask = poles['k']<attrs['knyq'].min()/3.
		mask = poles['k']>=0.
		k = poles['k'][mask]
		volume = attrs['kfun'].prod()*poles['modes'][mask]
		
		if key == 'A':
			def Wk(ell):
				return poles['{}_{:d}'.format(key,ell)][mask]/scipy.sqrt(self.normalization)
		if key == 'power':
			def Wk(ell):
				if ell == 0: return poles['{}_{:d}'.format(key,ell)].real[mask]/self.normalization-self.shotnoise
				return poles['{}_{:d}'.format(key,ell)].real[mask]/self.normalization
		
		window.window = scipy.empty((len(window.poles),len(window.s)),dtype=Wk(0).dtype)
		
		kk,ks = scipy.meshgrid(k,window.s,sparse=False,indexing='ij')
		ks = kk*ks
		for ill,ell in enumerate(window):
			self.logger.info('Hankel transforming {}_{:d}.'.format(key,ell))
			integrand = Wk(ell)[:,None]*1./(2.*constants.pi)**3*(-1)**(ell//2)*special.spherical_jn(ell,ks)
			window.window[ill] = scipy.sum(volume[:,None]*integrand,axis=0)

		window.norm = self.normref
		window.pad_zero()
		
		return window


class Real3PCFBinned(BaseCount):

	logger = logging.getLogger('Real3PCFBinned')
	
	@property
	def modes(self):
		if self.params.get('modes',None) is not None: return self.params['modes']
		if self.ic == 'global': return ['alpha','alpha','global']
		if self.ic == 'radial': return ['alpha','alpha','radial']
		if self.ic == 'angular': return ['alpha','alpha','angular']

	@utils.classparams
	def set_multipoles(self,icut=0,icut1=0,ncuts=1,losn=0,modes=None):
		
		catalogues = [Catalogue.load(self.params['path_randoms'][mode]) for mode in self.modes]
		
		icutibins = catalogues[1].attrs['icutibins'][icut]
		assert (icutibins == catalogues[2].attrs['icutibins'][icut]).all()
		icutnbins = len(icutibins)
		naddbins = catalogues[0].attrs['naddbins']
		assert (naddbins == catalogues[1].attrs['naddbins']) and (naddbins == catalogues[2].attrs['naddbins'])
		
		catalogue1 = catalogues[0].slice(icut1,ncuts)
		
		pyreal3pcf = PyReal3PCF(**self.params)
		pyreal3pcf.set_grid()

		self.result = 0
		for iaddbin in range(naddbins):
			mask2 = catalogues[1]['iaddbin'] == iaddbin
			mask3 = catalogues[2]['iaddbin'] == iaddbin
			for ibin,bin in enumerate(icutibins):
				self.logger.info('Correlating slice {:d} ({:d}/{:d}) of cut {:d}/{:d}.'.format(bin,ibin+1,icutnbins,iaddbin+1,naddbins))
				catalogue2 = catalogues[1][mask2 & (catalogues[1]['ibin'] == bin)]
				catalogue3 = catalogues[2][mask3 & (catalogues[2]['ibin'] == bin)]
				if not catalogue2 or not catalogue3: continue
				self.result += pyreal3pcf.copy().run(catalogue1,catalogue2,catalogue3)

	def to_window(self,**params):
	
		window = WindowFunction2D(**params)
		
		ells = self.result.ells
		sedges = self.result.sedges
		counts = self.result.counts
		
		window.s = [edges_to_mid(sedge) for sedge in sedges]
		window.poles = [(ell1,ell2) for ell1 in ells[0] for ell2 in ells[1]]
		window.los = self.result.los
		window.window = counts.reshape((-1,)+counts.shape[2:]) #window.window is (ell,s,d)
		if window.zero in window: window.error = window.window[window.index(window.zero)]**(1./3.)

		volume = (4.*constants.pi)**2*scipy.prod(scipy.meshgrid(*map(radial_volume,sedges),sparse=False,indexing='ij'),axis=0)
		for ill,(ell1,ell2) in enumerate(window): window.window[ill] *= (2*ell1+1)*(2*ell2+1)/volume

		window.window /= self.normalization
		if hasattr(window,'error'): window.error /= volume*self.normalization

		window.norm = self.normref
		window.pad_zero()

		return window

class Real3PCFBinnedShotNoise(Real3PCFBinned):

	logger = logging.getLogger('Real3PCFBinnedShotNoise')
	
	@utils.classparams
	def set_multipoles(self,icut=0,icut1=0,ncuts=1,modes=None):
		
		catalogues = [Catalogue.load(self.params['path_randoms'][mode]) for mode in self.modes]
		
		icutibins = catalogues[1].attrs['icutibins'][icut]
		assert (icutibins == catalogues[2].attrs['icutibins'][icut]).all()
		icutnbins = len(icutibins)
		naddbins = catalogues[0].attrs['naddbins']
		assert (naddbins == catalogues[1].attrs['naddbins']) and (naddbins == catalogues[2].attrs['naddbins'])
		assert scipy.allclose(catalogues[0]['Position'],catalogues[2]['Position'],rtol=1e-05,atol=1e-05)
		catalogues[0]['Weight'] *= catalogues[2]['Weight']
		
		catalogues[0] = catalogues[0].slice(icut1,ncuts)

		pyreal2pcf = PyReal2PCF(**self.params)
		pyreal2pcf.set_grid()
		
		self.result = 0
		for iaddbin in range(naddbins):
			mask1 = catalogues[0]['iaddbin'] == iaddbin
			mask2 = catalogues[1]['iaddbin'] == iaddbin
			for ibin,bin in enumerate(icutibins):
				self.logger.info('Correlating slice {:d} ({:d}/{:d}) of cut {:d}/{:d}.'.format(bin,ibin+1,icutnbins,iaddbin+1,naddbins))
				catalogue1 = catalogues[0][mask1 & (catalogues[0]['ibin'] == bin)]
				catalogue2 = catalogues[1][mask2 & (catalogues[1]['ibin'] == bin)]
				if not catalogue1 or not catalogue2: continue
				self.result += pyreal2pcf.copy().run(catalogue1,catalogue2)
		
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
		if window.zero in window: window.error = window.window[window.index(window.zero)]**(1./2.)

		volume = (4.*constants.pi)*radial_volume(sedges)
		for ill,ell in enumerate(window): window.window[ill] *= 2*(2*ell+1)/volume # legendre normalization and radial volume, 2 because we provide 2*W3

		window.window /= self.weight_tot
		if hasattr(window,'error'): window.error /= volume*self.weight_tot
		
		window.norm = 1.
		window.pad_zero()
		
		return window
	
class Real4PCFBinned(BaseCount):

	logger = logging.getLogger('Real4PCFBinned')

	@property
	def modes(self):
		if self.params.get('modes',None) is not None: return self.params['modes']
		if self.ic == 'radial': return ['alpha','alpha','radial','radial']
		if self.ic == 'angular': return ['alpha','alpha','angular','angular']
	
	@utils.classparams
	def set_multipoles(self,icut=0,losn=0,modes=None):
	
		catalogues = [Catalogue.load(self.params['path_randoms'][mode]) for mode in self.modes]

		icutibins = catalogues[0].attrs['icutibins'][icut]
		assert (icutibins == catalogues[2].attrs['icutibins'][icut]).all()
		icutnbins = len(icutibins)
		naddbins = catalogues[0].attrs['naddbins']
		assert (naddbins == catalogues[1].attrs['naddbins']) and (naddbins == catalogues[2].attrs['naddbins']) and (naddbins == catalogues[3].attrs['naddbins'])

		pyreal4pcf = PyReal4PCFBinned(**self.params)
		pyreal4pcf.set_grid(binsize=catalogues[-1].attrs['nbins'])

		self.result = 0
		for iaddbin1 in range(naddbins):
			catalogue2 = catalogues[1][catalogues[1]['iaddbin'] == iaddbin1]
			catalogue4 = catalogues[3][catalogues[3]['iaddbin'] == iaddbin1]
			for iaddbin2 in range(naddbins):
				mask1 = catalogues[0]['iaddbin'] == iaddbin2
				mask3 = catalogues[2]['iaddbin'] == iaddbin2
				for ibin,bin in enumerate(icutibins):
					self.logger.info('Correlating {}x{} slice {:d} ({:d}/{:d}) of cut {:d}/{:d} and {:d}/{:d}.'.format(self.modes[2],self.modes[3],
									bin,ibin+1,icutnbins,iaddbin1+1,naddbins,iaddbin2+1,naddbins))
					catalogue1 = catalogues[0][mask1 & (catalogues[0]['ibin'] == bin)]
					catalogue3 = catalogues[2][mask3 & (catalogues[2]['ibin'] == bin)]
					if not catalogue1 or not catalogue3: continue
					self.result += pyreal4pcf.copy().run(catalogue1,catalogue2,catalogue3,catalogue4,tobin=[2,4])
					if self.modes[2] != self.modes[3]:
					#if True:
						self.logger.info('Correlating {}x{} slice {:d} ({:d}/{:d}) of cut {:d}/{:d} and {:d}/{:d}.'.format(self.modes[3],self.modes[2],
									bin,ibin+1,icutnbins,iaddbin1+1,naddbins,iaddbin2+1,naddbins))
						self.result += pyreal4pcf.copy().run(catalogue2,catalogue1,catalogue4,catalogue3,tobin=[1,3]) # beware: twice the normalisation
		if self.result and self.modes[2] != self.modes[3]: self.result.weight_tot /= 2.
		#if True: self.result.weight_tot /= 2.
		
	def to_window(self,**params):

		window = WindowFunction2D(**params)
		
		ells = self.result.ells
		sedges = self.result.sedges
		counts = self.result.counts
		
		window.poles = [(ell1,ell2) for ell1 in ells[0] for ell2 in ells[1]]
		window.los = self.result.los
		window.s = [edges_to_mid(sedge) for sedge in sedges]
		window.window = counts.reshape((-1,)+counts.shape[2:])
		if window.zero in window: window.error = window.window[window.index(window.zero)]**(1./4.)

		volume = (4.*constants.pi)**2*scipy.prod(scipy.meshgrid(*map(radial_volume,sedges),sparse=False,indexing='ij'),axis=0)
		for ill,(ell1,ell2) in enumerate(window): window.window[ill] *= (2*ell1+1)*(2*ell2+1)/volume

		window.window /= self.normalization
		if hasattr(window,'error'): window.error /= volume*self.normalization
		
		window.norm = self.normref
		window.pad_zero()

		return window

class Real4PCFBinnedShotNoise(Real4PCFBinned):

	logger = logging.getLogger('Real4PCFBinnedShotNoise')
	
	@utils.classparams
	def set_multipoles(self,icut=0,modes=None):
		
		catalogues = [Catalogue.load(self.params['path_randoms'][mode]) for mode in self.modes]
		
		icutibins = catalogues[0].attrs['icutibins'][icut]
		assert (icutibins == catalogues[2].attrs['icutibins'][icut]).all()
		icutnbins = len(icutibins)
		naddbins = catalogues[0].attrs['naddbins']
		assert (naddbins == catalogues[1].attrs['naddbins']) and (naddbins == catalogues[2].attrs['naddbins']) and (naddbins == catalogues[3].attrs['naddbins'])
		assert scipy.allclose(catalogues[2]['Position'],catalogues[3]['Position'],rtol=1e-05,atol=1e-05)
		catalogues[2]['Weight'] *= catalogues[3]['Weight']

		self.result = 0
		
		if self.modes[2] == self.modes[3]:
			pyreal2pcf = PyReal2PCF(**self.params)
			pyreal2pcf.set_grid()
			for iaddbin in range(naddbins):
				mask1 = catalogues[0]['iaddbin'] == iaddbin
				mask2 = catalogues[1]['iaddbin'] == iaddbin
				mask3 = catalogues[2]['iaddbin'] == iaddbin
				for ibin,bin in enumerate(icutibins):
					self.logger.info('Correlating {}x{} slice {:d} ({:d}/{:d}) of cut {:d}/{:d}.'.format(self.modes[2],self.modes[3],
									bin,ibin+1,icutnbins,iaddbin+1,naddbins))
					catalogue1 = catalogues[0][mask1 & (catalogues[0]['ibin'] == bin)]
					catalogue2 = catalogues[1][mask2 & (catalogues[1]['ibin'] == bin)]
					if not catalogue1 or not catalogue2: continue
					weight = catalogues[2]['Weight'][mask3 & (catalogues[2]['ibin'] == bin)].sum()
					self.result += pyreal2pcf.copy().run(catalogue1,catalogue2)*weight
		else:
			pyreal2pcf = PyReal2PCFBinned(**self.params)
			pyreal2pcf.set_grid(binsize=catalogues[-1].attrs['nbins'])
			for iaddbin in range(naddbins):
				mask1 = catalogues[0]['iaddbin'] == iaddbin
				mask2 = catalogues[1]['iaddbin'] == iaddbin
				mask3 = catalogues[2]['iaddbin'] == iaddbin
				catalogue2 = catalogues[1][mask2]
				for ibin,bin in enumerate(icutibins):
					self.logger.info('Correlating {}x{} slice {:d} ({:d}/{:d}) of cut {:d}/{:d}.'.format(self.modes[2],self.modes[3],
									bin,ibin+1,icutnbins,iaddbin+1,naddbins,naddbins))
					catalogue1 = catalogues[0][mask1 & (catalogues[0]['ibin'] == bin)]
					if not catalogue1 or not catalogue2: continue
					mask = mask3 & (catalogues[2]['ibin'] == bin)
					weight = scipy.bincount(catalogues[3]['ibin'][mask],weights=catalogues[2]['Weight'][mask],minlength=catalogues[3].attrs['nbins'])
					self.result += pyreal2pcf.copy().run(catalogue1,catalogue2,tobin=2)*weight
					self.logger.info('Correlating {}x{} slice {:d} ({:d}/{:d}) of cut {:d}/{:d}.'.format(self.modes[3],self.modes[2],
									bin,ibin+1,icutnbins,iaddbin+1,naddbins))
					self.result += pyreal2pcf.copy().run(catalogue2,catalogue1,tobin=1)*weight
			if self.result: self.result.weight_tot /= 2.
		
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
		if window.zero in window: window.error = window.window[window.index(window.zero)]**(1./2.)

		volume = (4.*constants.pi)*radial_volume(sedges)
		for ill,ell in enumerate(window): window.window[ill] *= (2*ell+1)/volume #legendre normalization and radial volume

		window.window /= self.weight_tot
		if hasattr(window,'error'): window.error /= volume*self.weight_tot

		window.norm = 1.
		window.pad_zero()
		
		return window

def edges_to_mid(edges):
	#return scipy.sqrt((edges[1:]**3-edges[:-1]**3)/3./(edges[1:]-edges[:-1]))
	return 3./4.*(edges[1:]**4-edges[:-1]**4)/(edges[1:]**3-edges[:-1]**3)

def radial_volume(edges):
	return 1./3.*(edges[1:]**3-edges[:-1]**3)
