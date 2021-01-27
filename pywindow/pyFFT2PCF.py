import numpy
import logging
import time
import warnings

from nbodykit import CurrentMPIComm
from nbodykit.utils import timer
from nbodykit.meshtools import SlabIterator
from nbodykit.binned_statistic import BinnedStatistic
from pmesh.pm import ComplexField
	
def project_to_basis(y3d, edges, los=[0, 0, 1], poles=[]):
	"""
	Project a 3D statistic on to the specified basis. The basis will be one
	of:

	- 2D (`x`, `mu`) bins: `mu` is the cosine of the angle to the line-of-sight
	- 2D (`x`, `ell`) bins: `ell` is the multipole number, which specifies
	  the Legendre polynomial when weighting different `mu` bins

	.. note::

		The 2D (`x`, `mu`) bins will be computed only if `poles` is specified.
		See return types for further details.

	Notes
	-----
	*   the `mu` range extends from 0.0 to 1.0
	*   the `mu` bins are half-inclusive half-exclusive, except the last bin
		is inclusive on both ends (to include `mu = 1.0`)

	Parameters
	----------
	y3d : RealField or ComplexField
		the 3D array holding the statistic to be projected to the specified basis
	edges : list of arrays, (2,)
		list of arrays specifying the edges of the desired `x` bins and `mu` bins
	los : array_like,
		the line-of-sight direction to use, which `mu` is defined with
		respect to; default is [0, 0, 1] for z.
	poles : list of int, optional
		if provided, a list of integers specifying multipole numbers to
		project the 2d `(x, mu)` bins on to
	hermitian_symmetric : bool, optional
		Whether the input array `y3d` is Hermitian-symmetric, i.e., the negative
		frequency terms are just the complex conjugates of the corresponding
		positive-frequency terms; if ``True``, the positive frequency terms
		will be explicitly double-counted to account for this symmetry

	Returns
	-------
	result : tuple
		the 2D binned results; a tuple of ``(xmean_2d, mumean_2d, y2d, N_2d)``, where:

		- xmean_2d : array_like, (Nx, Nmu)
			the mean `x` value in each 2D bin
		- mumean_2d : array_like, (Nx, Nmu)
			the mean `mu` value in each 2D bin
		- y2d : array_like, (Nx, Nmu)
			the mean `y3d` value in each 2D bin
		- N_2d : array_like, (Nx, Nmu)
			the number of values averaged in each 2D bin

	pole_result : tuple or `None`
		the multipole results; if `poles` supplied it is a tuple of ``(xmean_1d, poles, N_1d)``,
		where:

		- xmean_1d : array_like, (Nx,)
			the mean `x` value in each 1D multipole bin
		- poles : array_like, (Nell, Nx)
			the mean multipoles value in each 1D bin
		- N_1d : array_like, (Nx,)
			the number of values averaged in each 1D bin
	"""
	comm = y3d.pm.comm
	x3d = y3d.x
	hermitian_symmetric = numpy.iscomplexobj(y3d)

	from scipy.special import legendre

	# setup the bin edges and number of bins
	xedges, muedges = edges
	x2edges = xedges**2
	Nx = len(xedges) - 1
	Nmu = len(muedges) - 1

	# always make sure first ell value is monopole, which
	# is just (x, mu) projection since legendre of ell=0 is 1
	do_poles = len(poles) > 0
	_poles = [0]+sorted(poles) if 0 not in poles else sorted(poles)
	legpoly = [legendre(l) for l in _poles]
	ell_idx = [_poles.index(l) for l in poles]
	Nell = len(_poles)

	# valid ell values
	if any(ell < 0 for ell in _poles):
		raise ValueError("in `project_to_basis`, multipole numbers must be non-negative integers")

	# initialize the binning arrays
	musum = numpy.zeros((Nx+2, Nmu+2))
	xsum = numpy.zeros((Nx+2, Nmu+2))
	ysum = numpy.zeros((Nell, Nx+2, Nmu+2), dtype=y3d.dtype) # extra dimension for multipoles
	Nsum = numpy.zeros((Nx+2, Nmu+2), dtype='i8')

	# if input array is Hermitian symmetric, only half of the last
	# axis is stored in `y3d`
	symmetry_axis = -1 if hermitian_symmetric else None

	# iterate over y-z planes of the coordinate mesh
	for slab in SlabIterator(x3d, axis=0, symmetry_axis=symmetry_axis):

		# the square of coordinate mesh norm
		# (either Fourier space k or configuraton space x)
		xslab = slab.norm2()

		# if empty, do nothing
		if len(xslab.flat) == 0: continue

		# get the bin indices for x on the slab
		dig_x = numpy.digitize(xslab.flat, x2edges)

		# make xslab just x
		xslab **= 0.5

		# get the bin indices for mu on the slab
		mu = slab.mu(los) # defined with respect to specified LOS
		dig_mu = numpy.digitize(abs(mu).flat, muedges)

		# make the multi-index
		multi_index = numpy.ravel_multi_index([dig_x, dig_mu], (Nx+2,Nmu+2))

		# sum up x in each bin (accounting for negative freqs)
		xslab[:] *= slab.hermitian_weights
		xsum.flat += numpy.bincount(multi_index, weights=xslab.flat, minlength=xsum.size)

		# count number of modes in each bin (accounting for negative freqs)
		Nslab = numpy.ones_like(xslab) * slab.hermitian_weights
		Nsum.flat += numpy.bincount(multi_index, weights=Nslab.flat, minlength=Nsum.size)

		# compute multipoles by weighting by Legendre(ell, mu)
		for iell, ell in enumerate(_poles):

			# weight the input 3D array by the appropriate Legendre polynomial
			weighted_y3d = legpoly[iell](mu) * y3d[slab.index]

			# add conjugate for this kx, ky, kz, corresponding to
			# the (-kx, -ky, -kz) --> need to make mu negative for conjugate
			# Below is identical to the sum of
			# Leg(ell)(+mu) * y3d[:, nonsingular]	(kx, ky, kz)
			# Leg(ell)(-mu) * y3d[:, nonsingular].conj()  (-kx, -ky, -kz)
			# or
			# weighted_y3d[:, nonsingular] += (-1)**ell * weighted_y3d[:, nonsingular].conj()
			# but numerically more accurate.
			if hermitian_symmetric:

				if ell % 2: # odd, real part cancels
					weighted_y3d.real[slab.nonsingular] = 0.
					weighted_y3d.imag[slab.nonsingular] *= 2.
				else:  # even, imag part cancels
					weighted_y3d.real[slab.nonsingular] *= 2.
					weighted_y3d.imag[slab.nonsingular] = 0.

			# sum up the weighted y in each bin
			weighted_y3d *= (2.*ell + 1.)
			ysum[iell,...].real.flat += numpy.bincount(multi_index, weights=weighted_y3d.real.flat, minlength=Nsum.size)
			if numpy.iscomplexobj(ysum):
				ysum[iell,...].imag.flat += numpy.bincount(multi_index, weights=weighted_y3d.imag.flat, minlength=Nsum.size)

		# sum up the absolute mag of mu in each bin (accounting for negative freqs)
		mu[:] *= slab.hermitian_weights
		musum.flat += numpy.bincount(multi_index, weights=abs(mu).flat, minlength=musum.size)

	# sum binning arrays across all ranks
	xsum  = comm.allreduce(xsum)
	musum = comm.allreduce(musum)
	ysum  = comm.allreduce(ysum)
	Nsum  = comm.allreduce(Nsum)

	# add the last 'internal' mu bin (mu == 1) to the last visible mu bin
	# this makes the last visible mu bin inclusive on both ends.
	ysum[..., -2] += ysum[..., -1]
	musum[:, -2]  += musum[:, -1]
	xsum[:, -2]   += xsum[:, -1]
	Nsum[:, -2]   += Nsum[:, -1]

	# reshape and slice to remove out of bounds points
	sl = slice(1, -1)
	with numpy.errstate(invalid='ignore'):

		# 2D binned results
		y2d	   = (ysum[0,...] / Nsum)[sl,sl] # ell=0 is first index
		xmean_2d  = (xsum / Nsum)[sl,sl]
		mumean_2d = (musum / Nsum)[sl, sl]
		N_2d	  = Nsum[sl,sl]

		# 1D multipole results (summing over mu (last) axis)
		if do_poles:
			N_1d	 = Nsum[sl,sl].sum(axis=-1)
			xmean_1d = xsum[sl,sl].sum(axis=-1) / N_1d
			poles	= ysum[:, sl,sl].sum(axis=-1) / N_1d
			poles	= poles[ell_idx,...]

	# return y(x,mu) + (possibly empty) multipoles
	result = (xmean_2d, mumean_2d, y2d, N_2d)
	pole_result = (xmean_1d, poles, N_1d) if do_poles else None
	return result, pole_result

def get_real_Ylm(l, m):
	"""
	Return a function that computes the real spherical
	harmonic of order (l,m)

	Parameters
	----------
	l : int
		the degree of the harmonic
	m : int
		the order of the harmonic; abs(m) <= l

	Returns
	-------
	Ylm : callable
		a function that takes 4 arguments: (xhat, yhat, zhat)
		unit-normalized Cartesian coordinates and returns the
		specified Ylm

	References
	----------
	https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
	"""
	import sympy as sp

	# make sure l,m are integers
	l = int(l); m = int(m)

	# the relevant cartesian and spherical symbols
	x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
	xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
	phi, theta = sp.symbols('phi theta')
	defs = [(sp.sin(phi), y/sp.sqrt(x**2+y**2)),
			(sp.cos(phi), x/sp.sqrt(x**2+y**2)),
			(sp.cos(theta), z/sp.sqrt(x**2+y**2+z**2))]

	# the normalization factors
	if m == 0:
		amp = sp.sqrt((2*l+1) / (4*numpy.pi))
	else:
		amp = sp.sqrt(2*(2*l+1) / (4*numpy.pi) * sp.factorial(l-abs(m)) / sp.factorial(l+abs(m)))

	# the cos(theta) dependence encoded by the associated Legendre poly
	expr = (-1)**m * sp.assoc_legendre(l, abs(m), sp.cos(theta))

	# the phi dependence
	if m < 0:
		expr *= sp.expand_trig(sp.sin(abs(m)*phi))
	elif m > 0:
		expr *= sp.expand_trig(sp.cos(m*phi))

	# simplify
	expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
	expr = amp * expr.expand().subs([(x/r, xhat), (y/r, yhat), (z/r, zhat)])
	Ylm = sp.lambdify((xhat,yhat,zhat), expr, 'numexpr')

	# attach some meta-data
	Ylm.expr = expr
	Ylm.l	= l
	Ylm.m	= m

	return Ylm
	
def define_cartesian_box(catalogue, position='Position', selection='Selection', BoxCenter=None, BoxSize=None, BoxPad=0.05, **kwargs):
	"""
	Internal function to put the CatalogSource in a
	Cartesian box.

	This function add two necessary attribues:

	#. :attr:`BoxSize` : array_like, (3,)
		if not provided, the BoxSize in each direction is computed from
		the maximum extent of the Cartesian coordinates of the :attr:`randoms`
		Source, with an optional, additional padding
	#. :attr:`BoxCenter`: array_like, (3,)
		the mean coordinate value in each direction; this is used to re-center
		the Cartesian coordinates of the :attr:`data` and :attr:`randoms`
		to the range of ``[-BoxSize/2, BoxSize/2]``
	"""
	from nbodykit.utils import get_data_bounds

	# compute the min/max of the position data
	pos, sel = catalogue.read([position, selection])
	pos_min, pos_max = get_data_bounds(pos, catalogue.comm, selection=sel)

	# used to center the data in the first cartesian quadrant
	delta = abs(pos_max - pos_min)
	catalogue.attrs['BoxCenter'] = BoxCenter
	if catalogue.attrs['BoxCenter'] is None: catalogue.attrs['BoxCenter'] = 0.5 * (pos_min + pos_max)

	# BoxSize is padded diff of min/max coordinates
	catalogue.attrs['BoxSize'] = BoxSize
	catalogue.attrs['BoxPad'] = BoxPad
	if catalogue.attrs['BoxSize'] is None:
		delta *= 1.0 + catalogue.attrs['BoxPad']
		catalogue.attrs['BoxSize'] = numpy.ceil(delta) # round up to nearest integer
	
	if (catalogue.attrs['BoxSize']<delta).any(): raise ValueError('BoxSize too small to contain all data.')		
	# log some info
	if catalogue.comm.rank == 0:
		catalogue.logger.info("BoxSize = %s" %str(catalogue.attrs['BoxSize']))
		catalogue.logger.info("cartesian coordinate range: %s : %s" %(str(pos_min), str(pos_max)))
		catalogue.logger.info("BoxCenter = %s" %str(catalogue.attrs['BoxCenter']))

def recenter_position(catalogue, position):
	"""
	The Position of the objects, re-centered on the mesh to
	the range ``[-BoxSize/2, BoxSize/2]``.
	This subtracts ``BoxCenter`` from :attr:`attrs` from the original
	position array.
	"""
	return catalogue[position] - catalogue.attrs['BoxCenter']

def get_mesh(catalogue,position='Position',weight='Weight',mesh={},**kwargs):
	
	kwargs.update(mesh)
	define_cartesian_box(catalogue,position=position,**kwargs)
	catalogue['_RecenteredPosition'] = recenter_position(catalogue,position=position)
	mesh = catalogue.to_mesh(position='_RecenteredPosition',weight=weight,**mesh)
	if mesh.comm.rank == 0: mesh.logger.info('k-range: {:.4g} - {:.4g}.'.format(2*numpy.pi/mesh.attrs['BoxSize'].min(),numpy.pi*mesh.attrs['Nmesh'].min()/mesh.attrs['BoxSize'].max()))
	return mesh


class PyFFT2PCF(object):
	"""
	Algorithm to compute power spectrum multipoles using FFTs
	for a data survey with non-trivial geometry.

	Due to the geometry, the estimator computes the true power spectrum
	convolved with the window function (FFT of the geometry).

	This estimator implemented in this class is described in detail in
	Hand et al. 2017 (arxiv:1704.02357). It uses the spherical harmonic
	addition theorem such that only :math:`2\ell+1` FFTs are required to
	compute each multipole. This differs from the implementation in
	Bianchi et al. and Scoccimarro et al., which requires
	:math:`(\ell+1)(\ell+2)/2` FFTs.

	Results are computed when the object is inititalized, and the result is
	stored in the :attr:`poles` attribute. Important meta-data computed
	during algorithm execution is stored in the :attr:`attrs` dict. See the
	documenation of :func:`~ConvolvedFFTPower.run`.

	.. note::
		A full tutorial on the class is available in the documentation
		:ref:`here <convpower>`.

	.. note::
		Cross correlations are only supported when the FKP weight column
		differs between the two mesh objects, i.e., the underlying ``data``
		and ``randoms`` must be the same. This allows users to compute
		the cross power spectrum of the same density field, weighted
		differently.

	Parameters
	----------
	first : CatalogMesh
		the first source to paint the data/randoms; FKPCatalog is automatically
		converted to a FKPCatalogMesh, using default painting parameters
	poles : list of int
		a list of integer multipole numbers ``ell`` to compute
	second : CatalogMesh, optional
		the second source to paint the data/randoms; cross correlations are
		only supported when the weight column differs between the two mesh
		objects, i.e., the underlying ``data`` and ``randoms`` must be the same!
	kmin : float, optional
		the edge of the first wavenumber bin; default is 0
	dk : float, optional
		the spacing in wavenumber to use; if not provided; the fundamental mode
		of the box is used
		
	References
	----------
	* Hand, Nick et al. `An optimal FFT-based anisotropic power spectrum estimator`, 2017
	* Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
	  MNRAS, 2015
	* Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
	"""
	logger = logging.getLogger('PyFFT2PCF')

	def __init__(self, first, poles,
					second=None,
					kmin=0.,
					dk=None):
		
		if second is None: second = first

		self.first = first
		self.second = second

		# grab comm from first source
		self.comm = first.comm

		# check for comm mismatch
		assert second.comm is first.comm, "communicator mismatch between input sources"

		# make a list of multipole numbers
		if numpy.isscalar(poles):
			poles = [poles]

		# store meta-data
		self.attrs = {}
		self.attrs['poles'] = poles
		self.attrs['dk'] = dk
		self.attrs['kmin'] = kmin

		# store BoxSize and BoxCenter from source
		self.attrs['Nmesh'] = self.first.attrs['Nmesh'].copy()
		self.attrs['BoxSize'] = self.first.attrs['BoxSize']
		self.attrs['BoxCenter'] = self.first.attrs['BoxCenter']
		self.attrs['kfun'] = 2*numpy.pi/self.attrs['BoxSize']
		self.attrs['knyq'] = numpy.pi*self.attrs['Nmesh']/self.attrs['BoxSize']

		# grab some mesh attrs, too
		self.attrs['mesh.resampler'] = self.first.attrs['resampler']
		self.attrs['mesh.interlaced'] = self.first.attrs['interlaced']

		# and run
		self.run()

	def run(self):
		"""
		Compute the power spectrum multipoles. This function does not return
		anything, but adds several attributes (see below).

		Attributes
		----------
		edges : array_like
			the edges of the wavenumber bins
		poles : :class:`~nbodykit.binned_statistic.BinnedStatistic`
			a BinnedStatistic object that behaves similar to a structured array, with
			fancy slicing and re-indexing; it holds the measured multipole
			results, as well as the number of modes (``modes``) and average
			wavenumbers values in each bin (``k``)
		attrs : dict
			dictionary holding input parameters and several important quantites
			computed during execution:

			#. data.N, randoms.N :
				the unweighted number of data and randoms objects
			#. data.W, randoms.W :
				the weighted number of data and randoms objects, using the
				column specified as the completeness weights
			#. alpha :
				the ratio of ``data.W`` to ``randoms.W``
			#. data.norm, randoms.norm :
				the normalization of the power spectrum, computed from either
				the "data" or "randoms" catalog (they should be similar).
				See equations 13 and 14 of arxiv:1312.4611.
			#. BoxSize :
				the size of the Cartesian box used to grid the data and
				randoms objects on a Cartesian mesh.

			For further details on the meta-data, see
			:ref:`the documentation <fkp-meta-data>`.
		"""
		pm = self.first.pm

		# setup the binning in k out to the minimum nyquist frequency
		dk = 2*numpy.pi/pm.BoxSize.min() if self.attrs['dk'] is None else self.attrs['dk']
		self.edges = numpy.arange(self.attrs['kmin'], numpy.pi*pm.Nmesh.min()/pm.BoxSize.max() + dk/2, dk)

		# measure the binned 1D multipoles in Fourier space
		poles = self._compute_multipoles()

		# set all the necessary results
		self.poles = BinnedStatistic(['k'], [self.edges], poles, fields_to_sum=['modes'], **self.attrs)


	def to_pkmu(self, wedges, max_ell):
		"""
		Invert the measured multipoles :math:`P_\ell(k)` into power
		spectrum wedges, :math:`P(k,\mu)`.

		Parameters
		----------
		mu_edges : array_like
			the edges of the :math:`\mu` bins
		max_ell : int
			the maximum multipole to use when computing the wedges;
			all even multipoles with :math:`ell` less than or equal
			to this number are included

		Returns
		-------
		pkmu : BinnedStatistic
			a data set holding the :math:`P(k,\mu)` wedges
		"""
		from scipy.special import legendre
		from scipy.integrate import quad

		def compute_coefficient(ell, mumin, mumax):
			"""
			Compute how much each multipole contributes to a given wedges.
			This returns:

			.. math::
				\frac{1}{\mu_{max} - \mu_{max}} \int_{\mu_{min}}^{\mu^{max}} \mathcal{L}_\ell(\mu)
			"""
			norm = 1.0 / (mumax - mumin)
			return norm * quad(lambda mu: legendre(ell)(mu), mumin, mumax)[0]

		# make sure we have all the poles measured
		ells = list(range(0, max_ell+1, 2))
		if any('power_%d' %ell not in self.poles for ell in ells):
			raise ValueError("measurements for ells=%s required if max_ell=%d" %(ells, max_ell))

		# new data array
		dtype = numpy.dtype([('power', 'c8'), ('k', 'f8'), ('mu', 'f8')])
		data = numpy.zeros((self.poles.shape[0], len(wedges)), dtype=dtype)

		# loop over each wedge
		for imu, mulims in enumerate(wedges):

			# add the contribution from each Pell
			for ell in ells:
				coeff = compute_coefficient(ell, *mulims)
				data['power'][:,imu] += coeff * self.poles['power_%d' %ell]

			data['k'][:,imu] = self.poles['k']
			data['mu'][:,imu] = numpy.ones(len(data))*0.5*(mulims[1]+mulims[0])

		dims = ['k','mu']
		mu_edges = numpy.array([mu[0] for mu in wedges] + [wedges[-1][-1]])
		edges = [self.poles.edges['k'], mu_edges]
		return BinnedStatistic(dims=dims, edges=edges, data=data, **self.attrs)

	def getstate(self):
		state = dict(edges=self.edges,
					 poles=self.poles.data,
					 attrs=self.attrs,
					 weight_tot=getattr(self,'weight_tot'))
		return state

	@CurrentMPIComm.enable
	def setstate(self,state,comm=None):
		state = comm.bcast(state)
		self.__dict__.update(state)
		self.comm = comm
		self.poles = BinnedStatistic(['k'], [self.edges], self.poles, fields_to_sum=['modes'])
		
	@classmethod
	@CurrentMPIComm.enable
	def loadstate(cls, state, comm=None):
		self = object.__new__(cls)
		self.setstate(state,comm=comm)
		return self

	def save(self, output):
		"""
		Save the :attr:`poles` result to a JSON file with name ``output``.
		"""
		import json
		from nbodykit.utils import JSONEncoder

		# only the master rank writes
		if self.comm.rank == 0:
			self.logger.info('measurement done; saving result to %s' % output)

			with open(output, 'w') as ff:
				json.dump(self.getstate(), ff, cls=JSONEncoder)
			
	@classmethod
	@CurrentMPIComm.enable
	def load(cls, filename, comm=None):
		"""
		Load a result from ``filename`` that has been saved to
		disk with :func:`save`.
		"""
		import json
		from nbodykit.utils import JSONDecoder
		if comm.rank == 0:
			with open(filename, 'r') as ff:
				state = json.load(ff, cls=JSONDecoder)
		else:
			state = None
		return cls.loadstate(state,comm=comm)
				
	def _compute_multipoles(self):
		"""
		Compute the window-convoled power spectrum multipoles, for a data set
		with non-trivial survey geometry.

		This estimator builds upon the work presented in Bianchi et al. 2015
		and Scoccimarro et al. 2015, but differs in the implementation. This
		class uses the spherical harmonic addition theorem such that
		only :math:`2\ell+1` FFTs are required per multipole, rather than the
		:math:`(\ell+1)(\ell+2)/2` FFTs in the implementation presented by
		Bianchi et al. and Scoccimarro et al.

		References
		----------
		* Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
		  MNRAS, 2015
		* Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
		"""
		# clear compensation from the actions
		for source in [self.first, self.second]:
			source.actions[:] = []; source.compensated = False
			assert len(source.actions) == 0

		# compute the compensations
		compensation = {}
		for name, mesh in zip(['first', 'second'], [self.first, self.second]):
			compensation[name] = get_compensation(mesh)
			if self.comm.rank == 0:
				if compensation[name] is not None:
					args = (compensation[name]['func'].__name__, name)
					self.logger.info("using compensation function %s for source '%s'" % args)
				else:
					self.logger.warning("no compensation applied for source '%s'" % name)

		rank = self.comm.rank
		pm   = self.first.pm

		# setup the 1D-binning
		muedges = numpy.linspace(0, 1, 2, endpoint=True)
		edges = [self.edges, muedges]

		# make a structured array to hold the results
		cols   = ['k'] + ['power_%d' %l for l in sorted(self.attrs['poles'])] + ['A_%d' %l for l in sorted(self.attrs['poles'])] + ['modes']
		dtype  = ['f8'] + ['c8']*len(self.attrs['poles'])*2 + ['i8']
		dtype  = numpy.dtype(list(zip(cols, dtype)))
		result = numpy.empty(len(self.edges)-1, dtype=dtype)

		# offset the box coordinate mesh ([-BoxSize/2, BoxSize]) back to
		# the original (x,y,z) coords
		offset = self.attrs['BoxCenter'] + 0.5*pm.BoxSize / pm.Nmesh

		# always need to compute ell=0
		poles = sorted(self.attrs['poles'])
		if 0 not in poles:
			poles = [0] + poles
		assert poles[0] == 0

		# spherical harmonic kernels (for ell > 0)
		Ylms = [[get_real_Ylm(l,m) for m in range(-l, l+1)] for l in poles[1:]]

		# paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
		rfield1 = self.first.paint(Nmesh=self.attrs['Nmesh'])

		vol_per_cell = (pm.BoxSize/pm.Nmesh).prod()/rfield1.attrs['num_per_cell'] #to compensate the default normalization by num_per_cell
		rfield1[:] /= vol_per_cell
		meta1 = rfield1.attrs.copy()
		if rank == 0: self.logger.info("%s painting of 'first' done" %self.first.window)

		# FFT 1st density field and apply the paintbrush window transfer kernel
		cfield = rfield1.r2c()
		if compensation['first'] is not None:
			cfield.apply(out=Ellipsis, **compensation['first'])
		if rank == 0: self.logger.info('ell = 0 done; 1 r2c completed')

		# monopole A0 is just the FFT of the FKP density field
		# NOTE: this holds FFT of density field #1
		volume = pm.BoxSize.prod()
		A0_1 = ComplexField(pm)
		A0_1[:] = cfield[:] * volume # normalize with a factor of volume

		# paint second mesh too?
		if self.first is not self.second:

			# paint the second field
			rfield2 = self.second.paint(Nmesh=self.attrs['Nmesh'],normalize=False)
			rfield2[:] /= vol_per_cell
			meta2 = rfield2.attrs.copy()
			if rank == 0: self.logger.info("%s painting of 'second' done" %self.second.window)

			# need monopole of second field
			if 0 in self.attrs['poles']:

				# FFT density field and apply the paintbrush window transfer kernel
				A0_2 = rfield2.r2c()
				A0_2[:] *= volume
				if compensation['second'] is not None: A0_2.apply(out=Ellipsis, **compensation['second'])
				
		else:
			rfield2 = rfield1
			meta2 = meta1

			# monopole of second field is first field
			if 0 in self.attrs['poles']:
				A0_2 = A0_1

		# save the painted density field #2 for later
		density2 = rfield2.copy()

		# initialize the memory holding the Aell terms for
		# higher multipoles (this holds sum of m for fixed ell)
		# NOTE: this will hold FFTs of density field #2
		Aell = ComplexField(pm)

		# the real-space grid
		xgrid = [xx.astype('f8') + offset[ii] for ii, xx in enumerate(density2.slabs.optx)]
		xnorm = numpy.sqrt(sum(xx**2 for xx in xgrid))
		xgrid = [x/xnorm for x in xgrid]

		# the Fourier-space grid
		kgrid = [kk.astype('f8') for kk in cfield.slabs.optx]
		knorm = numpy.sqrt(sum(kk**2 for kk in kgrid)); knorm[knorm==0.] = numpy.inf
		kgrid = [k/knorm for k in kgrid]

		# loop over the higher order multipoles (ell > 0)
		start = time.time()
		for iell, ell in enumerate(poles[1:]):

			# clear 2D workspace
			Aell[:] = 0.

			# iterate from m=-l to m=l and apply Ylm
			substart = time.time()
			for Ylm in Ylms[iell]:

				# reset the real-space mesh to the original density #2
				rfield2[:] = density2[:]

				# apply the config-space Ylm
				for islab, slab in enumerate(rfield2.slabs):
					slab[:] *= Ylm(xgrid[0][islab], xgrid[1][islab], xgrid[2][islab])

				# real to complex of field #2
				rfield2.r2c(out=cfield)

				# apply the Fourier-space Ylm
				for islab, slab in enumerate(cfield.slabs):
					slab[:] *= Ylm(kgrid[0][islab], kgrid[1][islab], kgrid[2][islab])

				# add to the total sum
				Aell[:] += cfield[:]

				# and this contribution to the total sum
				substop = time.time()
				if rank == 0:
					self.logger.debug("done term for Y(l=%d, m=%d) in %s" %(Ylm.l, Ylm.m, timer(substart, substop)))

			# apply the compensation transfer function
			if compensation['second'] is not None:
				Aell.apply(out=Ellipsis, **compensation['second'])

			# factor of 4*pi from spherical harmonic addition theorem + volume factor
			Aell[:] *= 4*numpy.pi*volume

			# log the total number of FFTs computed for each ell
			if rank == 0:
				args = (ell, len(Ylms[iell]))
				self.logger.info('ell = %d done; %s r2c completed' %args)
				
			# project on to 1d k-basis (averaging over mu=[0,1])
			proj_result, _ = project_to_basis(Aell, edges)
			result['A_%d' %ell][:] = numpy.squeeze(proj_result[2])

			# calculate the power spectrum multipoles, slab-by-slab to save memory
			# NOTE: this computes (A0 of field #1) * (Aell of field #2).conj()
			for islab in range(A0_1.shape[0]):
				Aell[islab,...] = A0_1[islab] * Aell[islab].conj()

			# project on to 1d k-basis (averaging over mu=[0,1])
			proj_result, _ = project_to_basis(Aell, edges)
			result['power_%d' %ell][:] = numpy.squeeze(proj_result[2])

		# summarize how long it took
		stop = time.time()
		if rank == 0:
			self.logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))

		# also compute ell=0
		if 0 in self.attrs['poles']:
		
			# the 1D monopole
			proj_result, _ = project_to_basis(A0_2, edges)
			result['A_0'][:] = numpy.squeeze(proj_result[2])

			# the 3D monopole
			for islab in range(A0_1.shape[0]):
				A0_1[islab,...] = A0_1[islab]*A0_2[islab].conj()

			# the 1D monopole
			proj_result, _ = project_to_basis(A0_1, edges)
			result['power_0'][:] = numpy.squeeze(proj_result[2])

		# save the number of modes and k
		result['k'][:] = numpy.squeeze(proj_result[0])
		result['modes'][:] = numpy.squeeze(proj_result[-1])

		# copy over any painting meta data
		if self.first is self.second:
			copy_meta(self.attrs, meta1)
		else:
			copy_meta(self.attrs, meta1, prefix='first')
			copy_meta(self.attrs, meta2, prefix='second')

		return result


def get_compensation(mesh):
	toret = None
	try:
		compensation = mesh._get_compensation()
		toret = {'func':compensation[0][1], 'kind':compensation[0][2]}
	except:
		pass
	return toret

def copy_meta(attrs, meta, prefix=""):
	if prefix:
		prefix += '.'
	for key in meta:
		if key.startswith('data.') or key.startswith('randoms.'):
			attrs[prefix+key] = meta[key]

