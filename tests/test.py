import os
import logging
import scipy
from scipy import special,integrate
from pywindow import *
from pywindow import plot

# Example with a fake catalog
def make_initial_catalogue(path):
	from pywindow.catalogue import sky_to_cartesian
	rng = scipy.random.RandomState(seed=42)
	rmin,rmax = 2000.,3000.
	size = 100000
	catalogue = Catalogue()
	distance = rng.uniform(2000.,3000.,size=size)
	ramin,ramax,decmin,decmax = 0.,30.,-15,15.
	u1,u2 = rng.uniform(size=(2,size))
	cmin = scipy.sin(scipy.deg2rad(decmin))
	cmax = scipy.sin(scipy.deg2rad(decmax))
	ra = ramin + u1*(ramax-ramin)
	dec = 90.-scipy.rad2deg(scipy.arccos(cmin+u2*(cmax-cmin)))
	catalogue['Position'] = sky_to_cartesian(distance,ra,dec)
	catalogue['Weight'] = catalogue.ones()
	decfrac = scipy.diff(scipy.sin(scipy.deg2rad([decmin,decmax])),axis=0)
	rafrac = scipy.diff(scipy.deg2rad([ramin,ramax]),axis=0)
	area = decfrac*rafrac
	catalogue['NZ'] = catalogue['Weight'].sum()/(area*(rmax-rmin))/distance**2
	catalogue.to_fits(path)

def prepare_catalogue(path_input,ic=None,ncuts=1,downsample=2.,**params):
	catalogue = Catalogue.from_fits(path_input)
	catalogue.attrs['norm'] = scipy.sum(catalogue['NZ']*catalogue['Weight'])/scipy.sum(catalogue['Weight']**2) # for a power normalisation I2, should be I2/sum(wdata**2)
	catalogue.logger.info('Box size: {:.4g} Mpc/h.'.format(catalogue.boxsize()))
	catalogue.logger.info('Shuffling.')
	catalogue.shuffle(seed=78987) # shuffling the catalogue to avoid selecting a peculiar subsample with further cuts
	catalogue = catalogue.downsample(factor=downsample,seed=42)
	path_randoms = params['path_randoms']['alpha']
	catalogue.save(path_randoms,keep=['Position','Weight'])
	rr = BaseCount(**params)
	# in additional_bins, you should put extra selections where the integral constraint is enforced, e.g. for the ELGs, a list of functions selecting the different chunk_z
	rr.prepare_catalogues(path_randoms,params['path_randoms'],ic=ic,additional_bins=[],ncuts=ncuts,share='flat')

# For an eBOSS-like catalog
def get_cosmo_eBOSS():
	from nbodykit.lab import cosmology
	cosmo_kwargs = dict(Omega_m=0.31,omega_b=0.022,h=0.676,sigma8=0.8,n_s=0.97,N_ur=2.0328,m_ncdm=[0.06])
	cosmo_kwargs['Omega0_b'] = cosmo_kwargs.pop('omega_b')/cosmo_kwargs['h']**2
	Omega0_m = cosmo_kwargs.pop('Omega_m')
	sigma8 = cosmo_kwargs.pop('sigma8')
	cosmo = cosmology.Cosmology(**cosmo_kwargs).match(Omega0_m=Omega0_m).match(sigma8=sigma8)
	return cosmo

def get_radial_edges(rwidth,drange):
	nbins = scipy.rint((dmax-dmin)/rwidth).astype(int)
	return scipy.linspace(dmin,dmax,nbins+1)

def prepare_catalogue_from_eBOSS(path_input,ic=None,ncuts=1,downsample=2.,**params):
	"""
	For a power normalisation I2, ``catalogue.attrs['norm']`` should be set to``I2/sum(wdata**2)``.
	This does not matter for the correlation function, since the normalisation of integral constraint window functions cancels with that of survey geometry.
	"""
	catalogue = Catalogue.from_fits(path_input)
	# catalog total weight
	catalogue['Weight'] = catalogue['WEIGHT_SYSTOT']*catalogue['WEIGHT_CP']*catalogue['WEIGHT_NOZ']*catalogue['WEIGHT_FKP']

	from nbodykit.lab import transform
	cosmo = get_cosmo_eBOSS()
	catalogue.logger.info('Omega0_m: {:.4g}.'.format(cosmo.Omega0_m))
	catalogue['Position'] = transform.SkyToCartesian(catalogue['RA'],catalogue['DEC'],catalogue['Z'],cosmo=cosmo,degrees=True).compute()
	catalogue.attrs['norm'] = 1. # for a power normalisation I2, should be I2/sum(wdata**2)

	catalogue.logger.info('Box size: {:.4g} Mpc/h.'.format(catalogue.boxsize()))

	catalogue.logger.info('Shuffling.')
	catalogue.shuffle(seed=78987) # shuffling the catalogue to avoid selecting a peculiar subsample with further cuts

	catalogue = catalogue.downsample(factor=downsample,seed=42)

	path_randoms = params['path_randoms']['alpha']
	catalogue.save(path_randoms,keep=['Position','Weight'])

	rr = BaseCount(**params)
	# in additional_bins, you should put extra selections where the integral constraint is enforced, e.g. for the ELGs, a list of functions selecting the different chunk_z
	rr.prepare_catalogues(path_randoms,params['path_randoms'],ic=ic,additional_bins=[],ncuts=ncuts,share='flat')


if __name__ == '__main__':

	setup_logging()

	todo = []
	todo += ['RR','RRRrad','RRRradRrad']
	todo += ['RRRradSN','RRRradRradSN']
	todo += ['windowRad','windowRadLS']
	todo += ['plot']

	base_dir = './files/'
	utils.mkdir(base_dir)
	os.chdir(base_dir)

	cat = 'std'
	if cat == 'eBOSS':
		path_input = 'eBOSS_LRG_clustering_NGC_v5.ran.fits' # this catalog should be in ./files/
		boxsize = 5000. # large enough to encompass all the sample
		rwidth = 50. # 2 Mpc/h bins for the radial IC is a fair choice, 50 for testing purposes
		cosmo = get_cosmo_eBOSS()
		dmin,dmax = cosmo.comoving_distance(zrange)
		downsample = 0.0001 # base downsampling of the random catalogue, very small for testing purposes
	else:
		path_input = 'catalogue.fits'
		make_initial_catalogue(path_input)
		boxsize = 5000. # large enough to encompass all the sample
		rwidth = 50. # 2 Mpc/h bins for the radial IC is a fair choice, 50 for testing purposes
		dmin,dmax = 2000.,3000.
		ncuts = 3 # to split calculation in ncuts parts, convenient to distribute job on many nodes
		downsample = 0.1 # base downsampling of the random catalogue, very small for testing purposes
	redges = get_radial_edges(rwidth,[dmin,dmax])
	ncuts = 3 # to split calculation in ncuts parts, convenient to distribute job on many nodes

	parameters = {}
	parameters['RR'] = {}
	parameters['RR']['sedges'] = scipy.linspace(0.,boxsize,int(boxsize)+1)
	parameters['RR']['ells'] = [0,2,4,5,6,8]
	parameters['RR']['los'] = 'midpoint'
	parameters['RR']['losn'] = 0 # wide-angle expansion order
	parameters['RR']['nthreads'] = 8
	parameters['RR']['binning'] = 'lin' #'log' #'lin'
	parameters['RR']['path_randoms'] = {'alpha':'randoms_RR.npy'}
	parameters['RR']['save'] = ['RR_{:d}.npy'.format(icut) for icut in range(ncuts)]
	parameters['RR']['save_window'] = 'window_RR.npy'

	parameters['RRRrad'] = {}
	parameters['RRRrad']['sedges'] = scipy.linspace(0.,boxsize,1001)
	parameters['RRRrad']['ells'] = [0,2,4,5,6,8]
	parameters['RRRrad']['los'] = ['midpoint','midpoint']
	parameters['RRRrad']['losn'] = [0,0]
	#parameters['RRRrad']['rwidth'] = rwidth
	parameters['RRRrad']['redges'] = redges
	parameters['RRRrad']['nthreads'] = 8
	parameters['RRRrad']['verbose'] = 'quiet'
	parameters['RRRrad']['path_randoms'] = {'alpha':'randoms_RRRrad.npy','radial':'randoms_RRRrad_radial.npy'}
	parameters['RRRrad']['save'] = ['RRRrad_{:d}.npy'.format(icut) for icut in range(ncuts)]
	parameters['RRRrad']['save_window'] = 'window_RRRrad.npy'

	parameters['RRRradRrad'] = {}
	parameters['RRRradRrad']['sedges'] = scipy.linspace(0.,boxsize,1001)
	parameters['RRRradRrad']['ells'] = [0,2,4,5,6,8]
	parameters['RRRradRrad']['los'] = ['midpoint','midpoint']
	parameters['RRRradRrad']['losn'] = [0,0]
	#parameters['RRRradRrad']['rwidth'] = rwidth
	parameters['RRRradRrad']['redges'] = redges
	parameters['RRRradRrad']['nthreads'] = 8
	parameters['RRRradRrad']['verbose'] = 'quiet'
	parameters['RRRradRrad']['path_randoms'] = {'alpha':'randoms_RRRradRrad.npy','radial':'randoms_RRRradRrad_radial.npy'}
	parameters['RRRradRrad']['save'] = ['RRRradRrad_{:d}.npy'.format(icut) for icut in range(ncuts)]
	parameters['RRRradRrad']['save_window'] = 'window_RRRradRrad.npy'

	parameters['RRRradSN'] = {}
	parameters['RRRradSN']['sedges'] = scipy.linspace(0.,boxsize,1001)
	parameters['RRRradSN']['ells'] = [0,2,4,5,6,8]
	parameters['RRRradSN']['los'] = 'midpoint'
	parameters['RRRradSN']['redges'] = redges
	parameters['RRRradSN']['nthreads'] = 8
	parameters['RRRradSN']['verbose'] = 'quiet'
	parameters['RRRradSN']['path_randoms'] = {'alpha':'randoms_RRRradSN.npy','radial':'randoms_RRRradSN_radial.npy'}
	parameters['RRRradSN']['save'] = ['RRRradSN_{:d}.npy'.format(icut) for icut in range(ncuts)]
	parameters['RRRradSN']['save_window'] = 'window_RRRradSN.npy'

	parameters['RRRradRradSN'] = {}
	parameters['RRRradRradSN']['sedges'] = scipy.linspace(0.,boxsize,1001)
	parameters['RRRradRradSN']['ells'] = [0,2,4,5,6,8]
	parameters['RRRradRradSN']['los'] = 'midpoint'
	parameters['RRRradRradSN']['redges'] = redges
	parameters['RRRradRradSN']['nthreads'] = 8
	parameters['RRRradRradSN']['verbose'] = 'quiet'
	parameters['RRRradRradSN']['path_randoms'] = {'alpha':'randoms_RRRradRradSN.npy','radial':'randoms_RRRradRradSN_radial.npy'}
	parameters['RRRradRradSN']['save'] = ['RRRradRradSN_{:d}.npy'.format(icut) for icut in range(ncuts)]
	parameters['RRRradRradSN']['save_window'] = 'window_RRRradSN.npy'

	# kernel of the first term in Eq. 2.20 of arXiv:1904.08851
	if 'RR' in todo:
		# to obtain RR pair counts
		params = parameters['RR']
		# save catalogue
		prepare_catalogue(path_input,ic=None,ncuts=ncuts,downsample=downsample,**params)
		# compute RR pair counts
		for icut in range(ncuts): # you can split the calculation on ncuts nodes
			rr = Real2PCF(**params)
			rr.set_multipoles(icut=icut,ncuts=ncuts)
			rr.save(save=params['save'][icut])
		# then build up the window function
		rr = 0
		for save in params['save']: rr += Real2PCF.load(save) # sum over all the pair counts
		rr.rebin(4) # you can rebin to another bin size
		window = rr.to_window()
		window.save(params['save_window'])

	# kernel of the second and third terms in Eq. 2.20 of arXiv:1904.08851
	if 'RRRrad' in todo:

		params = parameters['RRRrad']
		# save catalogue
		prepare_catalogue(path_input,ic='radial',ncuts=ncuts,downsample=downsample/2.,**params)
		# compute RRR 3pcf
		for icut in range(ncuts):
			rr = Real3PCFBinned(**params)
			rr.set_multipoles(icut=icut,icut1=0,ncuts=20) # you can downsample the catalogue by a factor ncuts because 3pcf calculation is a bit slow
			rr.save(save=params['save'][icut])
		# then build up the window function
		rr = 0
		for save in params['save']: rr += Real3PCFBinned.load(save)
		window = rr.to_window()
		window.save(params['save_window'])

	# kernel of the fourth term in Eq. 2.20 of arXiv:1904.08851
	if 'RRRradRrad' in todo:

		params = parameters['RRRradRrad']
		# save catalogue
		prepare_catalogue(path_input,ic='radial',ncuts=ncuts,downsample=downsample/4.,**params)
		# compute RRRR 4pcf
		params = parameters['RRRradRrad']
		for icut in range(ncuts):
			rr = Real4PCFBinned(**params)
			rr.set_multipoles(icut=icut)
			rr.save(save=params['save'][icut])
		# then build up the window function
		rr = 0
		for save in params['save']: rr += Real4PCFBinned.load(save)
		window = rr.to_window()
		window.save(params['save_window'])

	# now the shot noise for each of the integral constraint terms
	if 'RRRradSN' in todo:

		params = parameters['RRRradSN']
		# save catalogue
		prepare_catalogue(path_input,ic='radial',ncuts=ncuts,downsample=2.*downsample,**params)
		# compute RR 2pcf
		params = parameters['RRRradSN']
		for icut in range(ncuts):
			rr = Real3PCFBinnedShotNoise(**params)
			rr.set_multipoles(icut=icut)
			rr.save(save=params['save'][icut])
		# then build up the window function
		rr = 0
		for save in params['save']: rr += Real3PCFBinnedShotNoise.load(save)
		window = rr.to_window()
		window.save(params['save_window'])

	if 'RRRradRradSN' in todo:

		params = parameters['RRRradRradSN']
		# save catalogue
		prepare_catalogue(path_input,ic='radial',ncuts=ncuts,downsample=2.*downsample,**params)
		# compute RR 2pcf
		params = parameters['RRRradRradSN']
		for icut in range(ncuts):
			rr = Real4PCFBinnedShotNoise(**params)
			rr.set_multipoles(icut=icut)
			rr.save(save=params['save'][icut])
		# then build up the window function
		rr = 0
		for save in params['save']: rr += Real4PCFBinnedShotNoise.load(save)
		window = rr.to_window()
		window.save(params['save_window'])

	# trick: you can also compute windows with different random densities (to better sample small scales) and merge them using window_total |= window
	# finally, sum all all windows:
	path_window_rad = 'window_rad.npy'
	path_window_radSN = 'window_radSN.npy'
	if 'windowRad' in todo:
		window = WindowFunction.load(parameters['RRRrad']['save_window'])
		window -= WindowFunction.load(parameters['RRRradRrad']['save_window'])
		window.save(save=path_window_rad)

		window = WindowFunction.load(parameters['RRRradSN']['save_window'])
		window -= WindowFunction.load(parameters['RRRradRradSN']['save_window'])
		window.save(save=path_window_radSN)

	# for the Landy-Szalay estimator, one must divide window_rad and window_radSN by the window function RR
	path_window_rad_ls = 'window_rad_ls.npy'
	path_window_radSN_ls = 'window_radSN_ls.npy'
	if 'windowRadLS' in todo:
		mu = scipy.linspace(0.,1.,100)

		window = WindowFunction.load(path_window_rad)
		windowrr = WindowFunction.load(parameters['RR']['save_window'])
		wmurr = 0
		for ill,ell in enumerate(windowrr.ells): wmurr += windowrr(window.s[0],ell)[...,None]*special.legendre(ell)(mu)
		toret = WindowFunction.load(path_window_rad)
		toret.window[...] = 0.
		for ill1,ell1 in enumerate(window.ells[0]):
			wmu = 0
			for ill2,ell2 in enumerate(window.ells[1]): wmu += window.window[window.index((ell1,ell2))][...,None]*special.legendre(ell2)(mu)
			wmu /= wmurr[:,None,:]
			for ill2,ell2 in enumerate(window.ells[1]): toret.window[toret.index((ell1,ell2))] = (2.*ell2+1.)*integrate.trapz(wmu*special.legendre(ell2)(mu),x=mu,axis=-1)
		toret.window[scipy.isnan(toret.window) | scipy.isinf(toret.window)] = 0.
		toret.save(path_window_rad_ls)

		window = WindowFunction.load(path_window_radSN)
		windowrr = WindowFunction.load(parameters['RR']['save_window'])
		wmurr = 0
		for ill,ell in enumerate(windowrr.ells): wmurr += windowrr(window.s,ell)[...,None]*special.legendre(ell)(mu)
		toret = WindowFunction.load(path_window_radSN)
		toret.window[...] = 0.
		wmu = 0
		for ill2,ell2 in enumerate(window.ells): wmu += window.window[window.index(ell2)][...,None]*special.legendre(ell2)(mu)
		wmu /= wmurr
		for ill2,ell2 in enumerate(window.ells): toret.window[toret.index(ell2)] = (2.*ell2+1.)*integrate.trapz(wmu*special.legendre(ell2)(mu),x=mu,axis=-1)
		toret.window[scipy.isnan(toret.window) | scipy.isinf(toret.window)] = 0.
		toret.save(path_window_radSN_ls)

	# let's just check that the integration of the 2D window function over the first dimension yields the RR window function
	if 'plot' in todo:
		plot.plot_window_function_2d_projected(path_window_rad,parameters['RR']['save_window'],divide=False,scale='sloglin',title='Window function',path='projected_window.png')

	"""
	Power spectrum:
	a) for the window function effect alone, Eq. 2.10 of arXiv:1904.08851, see :class:`window_convolution.ConvolutionMultipole`, where you input path_window={0:'window_RR.npy'}.
	then you call :meth:`window_convolution.ConvolutionMultipole.convolve` on the correlation function ``Xil``::

		convolution = ConvolutionMultipole()
		convolution.set_window(path_window={0:'window_RR.npy'})
		convolution.set_grid(s=s,ellsin=ellsout,ellsout=ellsout)
		Xilc = convolution.convolve(Xil)

	b) for the integral constraint correction, Eq. 2.18 of arXiv:1904.08851, see :class:`integral_contraint.ConvolvedIntegralConstraint`, where you input path_window={0:'window_rad.npy','SN':'window_radSN.npy'}
	then you need to subtract :meth:`integral_contraint.ConvolvedIntegralConstraint.ic` and :attr:`integral_contraint.ConvolvedIntegralConstraint.real_shotnoise` multiplied by shotnoise from your correlation function ``Xil``.

		cic = ConvolvedIntegralConstraint()
		cic.set_window(path_window={0:'window_rad.npy','SN':'window_radSN.npy'})
		cic.set_grid(s=s,d=d,ellsin=ellsout,ellsout=ellsout)
		Xilcic = Xilc - cic.ic(Xil) - cic.real_shotnoise*shotnoise # final model correlation function, to be Hankel transformed back to power spectrum

	Correlation function:
	a) for the integral constraint correction, Eq. 2.18 of arXiv:1904.08851, see :class:`integral_contraint.ConvolvedIntegralConstraint`, where you input path_window={0:'window_rad_ls.npy','SN':'window_radSN_ls.npy'}
	then you need to subtract :meth:`integral_contraint.ConvolvedIntegralConstraint.ic` and :attr:`integral_contraint.ConvolvedIntegralConstraint.real_shotnoise` multiplied by shotnoise from your correlation function ``Xil``.

		cic = ConvolvedIntegralConstraint()
		cic.set_window(path_window={0:'window_rad_ls.npy','SN':'window_radSN_ls.npy'})
		cic.set_grid(s=s,d=d,ellsin=ellsout,ellsout=ellsout)
		Xilic = Xil - cic.ic(Xil) - cic.real_shotnoise*shotnoise # final model correlation function
	"""
