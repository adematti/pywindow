import os
import logging
import scipy
from scipy import constants
from astropy.io import fits
from . import utils

def distance(position,axis=-1):
	return scipy.sqrt((position**2).sum(axis=axis))

def cartesian_to_sky(position,wrap=True,degree=True):
	"""Transform cartesian coordinates into distance, RA, Dec.

	Parameters
	----------
	position : array of shape (N,3)
		position in cartesian coordinates.
	wrap : bool, optional
		whether to wrap ra into [0,2*pi]
	degree : bool, optional
		whether RA, Dec are in degree (True) or radian (False).

	Returns
	-------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.

	"""
	dist = distance(position)
	ra = scipy.arctan2(position[:,1],position[:,0])
	if wrap: ra %= 2.*constants.pi
	dec = scipy.arcsin(position[:,2]/dist)
	if degree: return dist,ra/constants.degree,dec/constants.degree
	return dist,ra,dec

def sky_to_cartesian(dist,ra,dec,degree=True,dtype=None):
	"""Transform distance, RA, Dec into cartesian coordinates.

	Parameters
	----------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.
	degree : bool
		whether RA, Dec are in degree (True) or radian (False).
	dtype : dtype, optional
		return array dtype.

	Returns
	-------
	position : array
		position in cartesian coordinates; of shape (len(dist),3).

	"""
	conversion = 1.
	if degree: conversion = constants.degree
	position = [None]*3
	cos_dec = scipy.cos(dec*conversion)
	position[0] = cos_dec*scipy.cos(ra*conversion)
	position[1] = cos_dec*scipy.sin(ra*conversion)
	position[2] = scipy.sin(dec*conversion)
	return (dist*scipy.asarray(position,dtype=dtype)).T

class Catalogue(object):

	logger = logging.getLogger('Catalogue')

	def __init__(self,columns={},fields=None,**attrs):

		self.columns = {}
		if fields is None: fields = columns.keys()
		for key in fields:
			self.columns[key] = scipy.asarray(columns[key])
		self.attrs = attrs

	@classmethod
	def from_fits(cls,path,ext=1):
		self = cls()
		self.logger.info('Loading catalogue {}.'.format(path))
		hdulist = fits.open(path,mode='readonly',memmap=True)
		columns = hdulist[ext].columns
		self = cls(hdulist[ext].data,fields=columns.names)
		return self

	@classmethod
	def from_nbodykit(cls,catalogue,fields=None,allgather=True,**kwargs):
		if fields is None:
			columns = {key: catalogue[key].compute() for key in catalogue}
		else:
			columns = {key: catalogue[key].compute() for key in fields}
		if allgather:
			columns = {key: scipy.concatenate(catalogue.comm.allgather(columns[key])) for key in columns}
		attrs = getattr(catalogue,'attrs',{})
		attrs.update(kwargs)
		return cls(columns=columns,**attrs)

	def to_nbodykit(self,fields=None):
	
		from nbodykit.base.catalog import CatalogSource
		from nbodykit import CurrentMPIComm
	
		comm = CurrentMPIComm.get()
		if comm.rank == 0:
			source = self
		else:
			source = None
		source = comm.bcast(source)

		# compute the size
		start = comm.rank * source.size // comm.size
		end = (comm.rank  + 1) * source.size // comm.size

		new = object.__new__(CatalogSource)
		new._size = end - start
		CatalogSource.__init__(new,comm=comm)
		for key in source.fields:
			new[key] = new.make_column(source[key])[start:end]
		new.attrs.update(source.attrs)

		return new

	def to_fits(self,path,fields=None,remove=[]):
		from astropy.table import Table
		if fields is None: fields = self.fields
		if remove:
			for rm in remove: fields.remove(rm)
		table = Table([self[field] for field in fields],names=fields)
		self.logger.info('Saving catalogue to {}.'.format(path))
		table.write(path,format='fits',overwrite=True)

	def shuffle(self,fields=None,seed=None):
		if fields is None: fields = self.fields
		rng = scipy.random.RandomState(seed=seed)
		indices = self.indices()
		rng.shuffle(indices)
		for key in fields: self[key] = self[key][indices]

	def indices(self):
		return scipy.arange(self.size)

	def slice(self,islice=0,nslices=1):
		size = len(self)
		return self[islice*size//nslices:(islice+1)*size//nslices]

	def downsample(self,factor=2.,rng=None,seed=None):
		if factor >= 1.: return self
		if rng is None: rng = scipy.random.RandomState(seed=seed)
		mask = factor >= rng.uniform(0.,1.,len(self))
		return self[mask]

	def distance(self,position='Position',axis=-1):
		return distance(self[position],axis=axis)
	
	def cartesian_to_sky(self,position='Position',wrap=True,degree=True):
		return cartesian_to_sky(position=self[position],wrap=wrap,degree=degree)
	
	def sky_to_cartesian(self,distance='distance',ra='RA',dec='DEC',degree=True,dtype=None):
		return sky_to_cartesian(distance=self[distance],ra=self[ra],dec=self[dec],degree=degree,dtype=dtype)
	
	def footprintsize(self,ra='RA',dec='DEC',position=None,degree=True):
		# WARNING: assums footprint does not cross RA = 0
		if position is not None:
			degree = False
			_,ra,dec = cartesian_to_sky(position=self[position],wrap=True,degree=degree)
		else:
			ra,dec = self[ra],self[dec]
		conversion = 1.
		if degree: conversion = constants.degree
		ra = (ra*conversion) % (2.*constants.pi)
		dec = dec*conversion
		ra,dec = scipy.array([ra.min(),ra.max()]),scipy.array([dec.min(),dec.max()])
		ra_degree,dec_degree = ra/constants.degree,dec/constants.degree
		self.logger.info('RA x DEC: [{:.1f}, {:.1f}] x [{:.1f}, {:.1f}].'.format(ra_degree.min(),ra_degree.max(),dec_degree.min(),dec_degree.max()))
		position = sky_to_cartesian([1.,1.],ra,dec,degree=False,dtype=scipy.float64)
		return distance(position[0]-position[1])
	
	def box(self,position='Position',axis=-1):
		axis = 0 if axis == -1 else -1
		return (self[position].min(axis=axis),self[position].max(axis=axis))

	def boxsize(self,position='Position',axis=-1):
		lbox = scipy.diff(self.box(position=position,axis=axis),axis=0)[0]
		return scipy.sqrt((lbox**2).sum(axis=0))

	def __getitem__(self,name):
		if isinstance(name,list) and isinstance(name[0],str):
			return [self[name_] for name_ in name]
		if isinstance(name,str):
			if name in self.fields:
				return self.columns[name]
			else:
				raise KeyError('There is no field {} in the data.'.format(name))
		else:
			new = self.deepcopy()
			new.columns = {field:self.columns[field][name] for field in self.fields}
			return new

	def __setitem__(self,name,item):
		if isinstance(name,list) and isinstance(name[0],str):
			for name_,item_ in zip(name,item):
				self.data[name_] = item_
		if isinstance(name,str):
			self.columns[name] = item
		else:
			for key in self.fields:
				self.columns[key][name] = item

	def __delitem__(self,name):
		del self.columns[name]

	def	__contains__(self,name):
		return name in self.columns

	def __iter__(self):
		for field in self.columns:
			yield field

	def __str__(self):
		return str(self.columns)

	def __len__(self):
		return len(self[self.fields[0]])

	@property
	def size(self):
		return len(self)

	def zeros(self,dtype=scipy.float64):
		return scipy.zeros(len(self),dtype=dtype)

	def ones(self,dtype=scipy.float64):
		return scipy.ones(len(self),dtype=dtype)

	def falses(self):
		return self.zeros(dtype=scipy.bool_)

	def trues(self):
		return self.ones(dtype=scipy.bool_)

	def nans(self):
		return self.ones()*scipy.nan

	@property
	def fields(self):
		return list(self.columns.keys())

	def remove(self,name):
		del self.columns[name]

	def __radd__(self,other):
		if other == 0: return self
		else: return self.__add__(other)

	def __add__(self,other):
		new = {}
		fields = [field for field in self.fields if field in other.fields]
		for field in fields:
			new[field] = scipy.concatenate([self[field],other[field]],axis=0)
		import copy
		attrs = copy.deepcopy(self.attrs)
		attrs.update(copy.deepcopy(other.attrs))
		return self.__class__(new,fields=fields,**attrs)

	def as_dict(self,fields=None):
		if fields is None: fields = self.fields
		return {field:self[field] for field in fields}

	def getstate(self,fields=None):
		return {'columns':self.as_dict(fields),'attrs':self.attrs}

	def setstate(self,state):
		self.__dict__.update(state)

	@classmethod
	def loadstate(cls,state):
		self = cls()
		self.setstate(state)
		return self

	def save(self,save,keep=None):
		pathdir = os.path.dirname(save)
		utils.mkdir(pathdir)
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,save))
		scipy.save(save,self.getstate(keep))

	@classmethod
	def load(cls,path):
		state = {}
		try:
			state = scipy.load(path,allow_pickle=True)[()]
		except IOError:
			raise IOError('Invalid path: {}.'.format(path))
		cls.logger.info('Loading {}: {}.'.format(cls.__name__,path))
		return cls.loadstate(state)

	def copy(self):
		return self.__class__.loadstate(self.getstate())

	def deepcopy(self):
		import copy
		return copy.deepcopy(self)
