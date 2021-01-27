import functools
import time
import os
import logging
import scipy
from scipy import constants,integrate
import matplotlib

def classparams(func):
	@functools.wraps(func)
	def wrapper(self,*args,**kwargs):
		varnames = list(func.__code__.co_varnames)[1:func.__code__.co_argcount]
		defaults = [] if func.__defaults__ is None else list(func.__defaults__)
		funcargs = {key:val for key,val in zip(varnames,args)}
		funcdefaults = {key:val for key,val in zip(varnames[-len(defaults):],defaults)}
		for key in funcargs:
			self.params[key] = funcargs[key]
		for key in kwargs:
			self.params[key] = kwargs[key]
		for key in funcdefaults:
			if key not in self.params: self.params[key] = funcdefaults[key]
		for key in varnames:
			if key not in self.params:
				raise ValueError('Argument {} not provided to {} instance.'.format(key,self.__class__.__name__))
		for key in self.params:
			if key in varnames: funcargs[key] = self.params[key]
		return func(self,**funcargs)
	return wrapper

def setstateclass(func):
	@functools.wraps(func)
	def wrapper(self,state):
		self.__dict__.update(state)
		return func(self,state)
	return wrapper

def getstateclass(func):
	@functools.wraps(func)
	def wrapper(self):
		state = {}
		if hasattr(self,'params'): state['params'] = self.params
		return func(self,state)
	return wrapper

def loadstateclass(func):
	@functools.wraps(func)
	def wrapper(cls,state,**kwargs):
		self = object.__new__(cls)
		if kwargs:
			if 'params' in state: state['params'].update(kwargs)
			else: raise ValueError('Too many arguments for a class that does not contain params.')
		return func(self,state)
	return wrapper

def loadclass(func):
	@functools.wraps(func)
	def wrapper(cls,path,**kwargs):
		state = {}
		try:
			state = scipy.load(path,allow_pickle=True)[()]
		except IOError:
			raise IOError('Invalid path: {}.'.format(path))
		cls.logger.info('Loading {}: {}.'.format(cls.__name__,path))
		return loadstateclass(func)(cls,state,**kwargs)
	return wrapper

def saveclass(func):
	@functools.wraps(func)
	@classparams
	def wrapper(self,save=None,**kwargs):
		state = func(self)
		pathdir = os.path.dirname(save)
		mkdir(pathdir)
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,save))
		scipy.save(save,state)
	return wrapper

def save(path,*args,**kwargs):
	mkdir(os.path.dirname(path))
	logger.info('Saving to {}.'.format(path))
	scipy.save(path,*args,**kwargs)

def load(path,allow_pickle=True,**kwargs):
	logger.info('Loading {}.'.format(path))
	try:
		return scipy.load(path,allow_pickle=allow_pickle,**kwargs)[()]
	except IOError:
		raise IOError('Invalid path: {}.'.format(path))
"""
def mkdir(path):
	path = os.path.abspath(path)
	if not os.path.isdir(path): os.makedirs(path)
"""
def mkdir(path):
	try: os.makedirs(path) #MPI...
	except OSError: return

def savefig(path,*args,**kwargs):
	mkdir(os.path.dirname(path))
	logger.info('Saving figure to {}.'.format(path))
	matplotlib.pyplot.savefig(path,*args,**kwargs)
	matplotlib.pyplot.close(matplotlib.pyplot.gcf())

def tolist(el,n=None,fill=None,value=None):
	if not isinstance(el,list): el = [el]
	if n is None: return el
	miss = n-len(el)
	if isinstance(fill,int): el += [el[fill]]*miss
	else: el += [value]*miss
	return el

def fill_axis(a,axis=(0),slices=(0),values=0.):
	slices_ = [slice(None) for i in range(a.ndim)]
	for ax,sl in zip(axis,slices): slices_[ax] = sl
	a[tuple(slices_)] = values

def text_to_latex(txt):
	if txt in [0,2,4]: return '$\\ell = {:d}$'.format(txt)
	if txt == 'pk_sigma3sq': return '$\\sigma_{3}^{2}P_{\\mathrm{m}}^{\\mathrm{lin}}$'
	if txt[:3] == 'pk_':
		tmp = txt[3:].replace('d','ddd').replace('t','ttt')
		if 'b' in tmp:
			return '$P_{{ {0} }}$'.format(tmp.replace('ddd',',\\delta').replace('ttt',',\\theta'))
		else: 
			return '$P_{{ {0} }}$'.format(tmp.replace('ddd','\\delta').replace('ttt','\\theta'))

def sorted_parameters(args,return_sort=False,keys=['qpar','qper','qiso','eps','f','b1','b2','A','sigmav','sigma8']):
	indices = scipy.full(len(args),scipy.inf)
	for iarg,arg in enumerate(args):
		for ipar,par in enumerate(keys):
			if par in arg: indices[iarg] = ipar
	sort = scipy.argsort(indices)
	if return_sort: return scipy.array(args)[sort],sort
	return scipy.array(args)[sort].tolist()

def overlap(a,b):
	"""Returns the indices for which a and b overlap.
	Warning: makes sense if and only if a and b elements are unique.
	Taken from https://www.followthesheep.com/?p=1366.
	"""
	a=scipy.asarray(a)
	b=scipy.asarray(b)
	a1=scipy.argsort(a)
	b1=scipy.argsort(b)
	# use searchsorted:
	sort_left_a=a[a1].searchsorted(b[b1], side='left')
	sort_right_a=a[a1].searchsorted(b[b1], side='right')
	#
	sort_left_b=b[b1].searchsorted(a[a1], side='left')
	sort_right_b=b[b1].searchsorted(a[a1], side='right')

	# # which values are in b but not in a?
	# inds_b=(sort_right_a-sort_left_a==0).nonzero()[0]
	# # which values are in b but not in a?
	# inds_a=(sort_right_b-sort_left_b==0).nonzero()[0]

	# which values of b are also in a?
	inds_b=(sort_right_a-sort_left_a > 0).nonzero()[0]
	# which values of a are also in b?
	inds_a=(sort_right_b-sort_left_b > 0).nonzero()[0]

	return a1[inds_a], b1[inds_b]

def bin_ndarray(ndarray, new_shape, weights=None, operation=scipy.sum):
	"""Bin an ndarray in all axes based on the target shape, by summing or
	averaging. Number of output dimensions must match number of input dimensions and 
	new axes must divide old ones.

	Taken from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
	and https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/binned_statistic.html#BinnedStatistic.reindex.

	Example
	-------
	>>> m = np.arange(0,100,1).reshape((10,10))
	>>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
	>>> print(n)

	[[ 22  30  38  46  54]
	 [102 110 118 126 134]
	 [182 190 198 206 214]
	 [262 270 278 286 294]
	 [342 350 358 366 374]]

	"""
	if ndarray.ndim != len(new_shape):
		raise ValueError('Shape mismatch: {} -> {}'.format(ndarray.shape,new_shape))
	if any([c % d != 0 for d,c in zip(new_shape,ndarray.shape)]):
		raise ValueError('New shape must be a divider of the original shape'.format(ndarray.shape,new_shape))
	compression_pairs = [(d, c//d) for d,c in zip(new_shape,ndarray.shape)]
	flattened = [l for p in compression_pairs for l in p]
	ndarray = ndarray.reshape(flattened)
	if weights is not None: weights = weights.reshape(flattened)

	for i in range(len(new_shape)):
		if weights is not None:
			ndarray = operation(ndarray*weights, axis=-1*(i+1))
			weights = scipy.sum(weights, axis=-1*(i+1))
			ndarray /= weights
		else:
			ndarray = operation(ndarray, axis=-1*(i+1))

	return ndarray

def plot_xlabel(estimator='spectrum'):
	if estimator == 'spectrum':
		return '$k$ [$h \ \\mathrm{Mpc}^{-1}$]'
	if estimator == 'rmu':
		return '$r$ [$\\mathrm{Mpc} \ h^{-1}$]'
	if 'angular' in estimator:
		return '$\\theta$ [deg]'
	if  estimator == 'rppi':
		return '$r_{p}$ [$\\mathrm{Mpc} \ h^{-1}$]'
	if estimator == 'rr':
		return '$s$ [$\\mathrm{Mpc} \ h^{-1}$]'

def suplabel(axis,label,shift=0,labelpad=5,ha='center',va='center',**kwargs):
	"""Add super ylabel or xlabel to the figure. Similar to matplotlib.suptitle.
	Taken from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots.

	Parameters
	----------
	axis : str
		'x' or 'y'.
	label : str
		label.
	shift : float, optional
		shift.
	labelpad : float, optional
		padding from the axis.
	ha : str, optional
		horizontal alignment.
	va : str, optional
		vertical alignment.
	kwargs : dict
		kwargs for matplotlib.pyplot.text

	"""
	fig = matplotlib.pyplot.gcf()
	xmin = []
	ymin = []
	for ax in fig.axes:
		xmin.append(ax.get_position().xmin)
		ymin.append(ax.get_position().ymin)
	xmin,ymin = min(xmin),min(ymin)
	dpi = fig.dpi
	if axis.lower() == 'y':
		rotation = 90.
		x = xmin - float(labelpad)/dpi
		y = 0.5 + shift
	elif axis.lower() == 'x':
		rotation = 0.
		x = 0.5 + shift
		y = ymin - float(labelpad)/dpi
	else:
		raise Exception('Unexpected axis: x or y')
	matplotlib.pyplot.text(x,y,label,rotation=rotation,transform=fig.transFigure,ha=ha,va=va,**kwargs)

def mkdir(path):
	path = os.path.abspath(path)
	if not os.path.isdir(path): os.makedirs(path)

def savefig(path,*args,**kwargs):
	mkdir(os.path.dirname(path))
	logger.info('Saving figure to {}.'.format(path))
	matplotlib.pyplot.savefig(path,*args,**kwargs)
	matplotlib.pyplot.close(matplotlib.pyplot.gcf())


_logging_handler = None

def setup_logging(log_level="info"):
	"""
	Turn on logging, with the specified level.
	Taken from nbodykit: https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py.

	Parameters
	----------
	log_level : 'info', 'debug', 'warning'
		the logging level to set; logging below this level is ignored
	"""

	# This gives:
	#
	# [ 000000.43 ]   0: 06-28 14:49  measurestats	INFO	 Nproc = [2, 1, 1]
	# [ 000000.43 ]   0: 06-28 14:49  measurestats	INFO	 Rmax = 120

	levels = {
			"info" : logging.INFO,
			"debug" : logging.DEBUG,
			"warning" : logging.WARNING,
			}

	logger = logging.getLogger();
	t0 = time.time()


	class Formatter(logging.Formatter):
		def format(self, record):
			s1 = ('[ %09.2f ]: ' % (time.time() - t0))
			return s1 + logging.Formatter.format(self, record)

	fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
					datefmt='%m-%d %H:%M ')

	global _logging_handler
	if _logging_handler is None:
		_logging_handler = logging.StreamHandler()
		logger.addHandler(_logging_handler)

	_logging_handler.setFormatter(fmt)
	logger.setLevel(levels[log_level])

#setup_logging()
logger = logging.getLogger('Utils')
