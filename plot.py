import os
import scipy
from scipy import integrate,constants
import logging
from matplotlib import pyplot
from matplotlib.ticker import AutoMinorLocator,LogLocator,FixedLocator,NullFormatter
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib import gridspec
from .mu_function import MuFunction
from .window_function import WindowFunction
from . import utils

logger = logging.getLogger('Plot')

fontsize = 20
labelsize = 16
figsize = (8,6)
dpi = 200
prop_cycle = pyplot.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#linestyles = [(0,()),(0,(1,5)),(0,(5,5)),(0,(3,5,1,5)),(0,(3,5,1,5,1,5))]
linestyles = ['-','--',':','-.']
scalings = {}
scalings['rloglin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\\xi_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['rlogr2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\\xi_{{\\ell}}(s) [(\\mathrm{Mpc} \ h^{-1})^{2}]$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['rlinr2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\\xi_{{\\ell}}(s) [(\\mathrm{Mpc} \ h^{-1})^{2}]$','xscale':'linear','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slinlin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\mathcal{W}_{{\\ell}}(s)$','xscale':'linear','yscale':'linear','xlim':[1e-3,200.]}
scalings['sloglin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slogslin'] = {'func':lambda x,y: (x,x*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slogs2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slogs3lin'] = {'func':lambda x,y: (x,x**3*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{3}\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['sloglog'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'log','xlim':[1e-3,3000.]}
scalings['slogs2log'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'log','xlim':[1e-3,3000.]}

def sort_legend(ax,**kwargs):
	handles,labels = ax.get_legend_handles_labels()
	labels,handles = zip(*sorted(zip(labels,handles),key=lambda t: utils.str_to_int(t[0])[-1]))
	ax.legend(handles,labels,**kwargs)

def text_to_latex(txt):
	if txt in range(0,20,1): return '$\\ell = {:d}$'.format(txt)

def plot_baseline(scaling,title=''):
	fig = pyplot.figure(figsize=figsize)
	if title: fig.suptitle(title,fontsize=fontsize)
	ax = pyplot.gca()
	if scaling['xlim']: ax.set_xlim(scaling['xlim'])
	ax.set_xscale(scaling['xscale'])
	ax.set_yscale(scaling['yscale'])
	ax.grid(True)
	ax.tick_params(labelsize=labelsize)
	ax.set_xlabel(scaling['xlabel'],fontsize=fontsize)
	ax.set_ylabel(scaling['ylabel'],fontsize=fontsize)
	return ax

def plot_mu_function(path_window,scale='klinlin',title='$\\mu$ function',path='window_KMU.png'):

	window = MuFunction.load(path_window)
	k = window.k
	mu = window.mu
	#k = [0.,0.01]
	scaling = scalings[scale]
	scaling['xlim'] = None
	scaling['ylabel'] = '$\\mu$'
	xextend = 0.8
	ax = plot_baseline(scaling,title)
	#norm = Normalize(*[0.,1.])
	tmp = window.window
	norm = Normalize(tmp.min(),tmp.max())
	im = ax.pcolormesh(k,mu,window(k,mu).T,norm=norm,cmap=cm.jet_r)
	fig = pyplot.gcf()
	fig.subplots_adjust(right=xextend)
	cbar_ax = fig.add_axes([xextend+0.05,0.15,0.03,0.7])
	cbar_ax.tick_params(labelsize=labelsize) 
	cbar = fig.colorbar(im,cax=cbar_ax)

	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_function_1d(path_window,key,scale='sloglin',title='$\\mathcal{W}_{\ell}(s)$',path='window_1d.png',error=False):
	
	window = WindowFunction.load(path_window)
	s = window.s[window.s>0]
	#s = scipy.logspace(-1,5,3000,base=10)
	#s = s[(s>1000.) & (s<2000.)]
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(s[0],s[-1])
	
	for ill,ell in enumerate(window):
		if error and (ell==0): ax.errorbar(*scaling['func'](s,window(s,ell)),yerr=scaling['func'](s,window.poisson_error(s))[1],label=text_to_latex(ell),linestyle='-')
		else: ax.plot(*scaling['func'](s,window(s,ell)),label=text_to_latex(ell),linestyle='-')

	ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
	if losn % 2 != 1: sort_legend(ax,**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	else: sort_legend(ax,**{'loc':2,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	#ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)


def plot_window_function_2d(path_window,key,scale='slinlin',title='$\\mathcal{W}_{\ell}(s)$',path='window_2d.png'):
	
	window = WindowFunction.load(path_window)
	s,d = window.s[0][window.s[0]>0.],window.s[1][window.s[1]>0.]
	#s,d = window.s[0][window.s[0]>0.],window.s[1][window.s[1]>1.e3]
	scaling = scalings[scale]
	s = s[(s>=scaling['xlim'][0]) & (s<scaling['xlim'][-1])]
	d = d[(d>=scaling['xlim'][0]) & (d<scaling['xlim'][-1])]
	figsize = 7; xextend = 0.8
	xlabel,ylabel = ('$s$ [$\\mathrm{Mpc} \ h^{-1}$]','$\\Delta$ [$\\mathrm{Mpc} \ h^{-1}$]')
	cmap = pyplot.get_cmap('Spectral')
	ells1,ells2 = window.ells
	#ells1,ells2 = ells1[:3],ells2[:3]
	ells1,ells2 = [2],[0]
	tmp = [window((s,d),(ell1,ell2)) for ell1 in ells1 for ell2 in ells2]
	norm = Normalize(scipy.amin(tmp),scipy.amax(tmp))
	#norm = Normalize(0,1)
	ncols = len(ells1); nrows = len(ells2)
	fig = pyplot.figure(figsize=(figsize/xextend,figsize))
	if title is not None: fig.suptitle(title,fontsize=fontsize)
	gs = gridspec.GridSpec(nrows,ncols,wspace=0.1,hspace=0.1)
	for ill1,ell1 in enumerate(ells1):
		for ill2,ell2 in enumerate(ells2):
			ax = pyplot.subplot(gs[nrows-1-ill2,ill1])
			im = ax.pcolormesh(s,d,window([s,d],(ell1,ell2)).T,norm=norm,cmap=cmap)
			#im = ax.pcolormesh(s,d,scipy.log(scipy.absolute(window([s,d],(ell1,ell2)).T)),norm=norm,cmap=cmap)
			ax.set_xscale(scaling['xscale'])
			ax.set_yscale(scaling['xscale'])
			if ill1>0: ax.get_yaxis().set_visible(False)
			elif 'log' in scale:
				ax.yaxis.set_major_locator(LogLocator(base=10.,subs=(1,),numticks=3))
				ax.yaxis.set_minor_locator(LogLocator(base=10.,subs=range(1,11),numticks=3))
				ax.yaxis.set_minor_formatter(NullFormatter())
			if ill2>0: ax.get_xaxis().set_visible(False)
			elif 'log' in scale:
				ax.xaxis.set_major_locator(LogLocator(base=10.,subs=(1,),numticks=3))
				ax.xaxis.set_minor_locator(LogLocator(base=10.,subs=range(1,11),numticks=3))
				ax.xaxis.set_minor_formatter(NullFormatter())
			ax.tick_params(labelsize=labelsize)
			text = '$\\mathcal{{W}}_{{{:d},{:d}}}$'.format(ell1,ell2)
			ax.text(0.05,0.95,text,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='black',fontsize=labelsize)
	utils.suplabel('x',xlabel,shift=-0.04,labelpad=7,size=labelsize)
	utils.suplabel('y',ylabel,shift=0,labelpad=8,size=labelsize)
	fig.subplots_adjust(right=xextend)
	cbar_ax = fig.add_axes([xextend+0.05,0.15,0.03,0.7])
	cbar_ax.tick_params(labelsize=labelsize)
	cbar = fig.colorbar(im,cax=cbar_ax,format='%.1e')

	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
		

def plot_window_function_comparison(list_path_window,scale='sloglin',title='Window function',path='window_comparison.png'):
	

	window = WindowFunction.load(list_path_window[0])
	s = window.s[window.s>0]
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(s[0]*0.9,s[-1])

	for iwin,path_window in enumerate(list_path_window):
		window = WindowFunction.load(path_window)
		for ill,ell in enumerate(window):
			ax.plot(*scaling['func'](s,window(s,ell)),label=text_to_latex(ell) if iwin == 0 else None,linestyle=linestyles[iwin],color=colors[ill])

	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_function_shotnoise(path_window,path_window_rrr=None,scale='sloglin',title='$\\mathcal{W}_{\ell}(s)$',path='window_shotnoise.png'):
	
	window = WindowFunction.load(path_window)
	s = window.s[window.s>1.]
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(s[0]*0.9,s[-1])
	
	for ill,ell in enumerate(window):
		ax.plot(*scaling['func'](s,window(s,ell)),label=text_to_latex(ell),linestyle='-',color=colors[ill])
	
	if path_window_rrr is not None:
		windowrrr = WindowFunction.load(path_window_rrr)
		ax.plot(*scaling['func'](s,windowrrr([s,0.],(ell,0))),linestyle='--',color=colors[ill])

	sort_legend(ax,**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	#ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})

	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_function_2d_projected(path_window,path_window_1d,divide=False,scale='sloglin',title='Window function',path='projected_window.png'):
	
	window = WindowFunction.load(path_window)
	s,d = window.s[0][window.s[0]>0.],window.s[1][window.s[1]>0.]
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(d[0]*0.9,d[-1])
	windowrr = WindowFunction.load(path_window_1d)

	for ill,ell in enumerate(window.ells[-1]):
		proj = 4.*constants.pi*integrate.trapz(window([s,d],(0,ell))*s[:,None]**2,x=s,axis=0)
		if divide: proj /= 2. # double los
		ax.plot(*scaling['func'](d,proj),label=text_to_latex(ell),linestyle='-',color=colors[ill])
		ax.plot(*scaling['func'](d,windowrr(d,ell)),linestyle='--',color=colors[ill])

	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
