import logging
import scipy
from scipy import constants,integrate
from pywindow import WindowFunction,utils
from .window_convolution import MultipoleToMultipole

def weights_trapz(x):
	trapzw = 4.*constants.pi*x**2*scipy.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.
	return trapzw

def integrate_trapz(func,x):
	return scipy.sum(func*weights_trapz(x),axis=-1)

class BaseIntegralConstraint(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('IC')
	
	def __init__(self,**params):
		self.params = params
		self.window = {}

	@utils.classparams
	def set_window(self,path_window):
		# path_window is a dict which can contain shot noise with key 'SN', and window functions at order n in wide-angle expansion with key n.
		self.window = {key: WindowFunction.load(path_window[key]) for key in path_window}
		self.ns = sorted([key for key in self.window if key != 'SN'])
		assert self.ns[0] == 0
		self.los = self.window[0].los[-1][0]
		for n in self.ns: assert self.window[n].los[-1] == (self.los,n)

		return self.los, self.ns
	
	@property
	def ellsin(self):
		return self.params['ellsin']

	@property
	def ellsconv(self):
		return self.params['ellsconv']
	
	@property
	def ellsout(self):
		return self.params['ellsout']
		
	def indexin(self,ell):
		return self.ellsin.index(ell)

	def indexout(self,ell):
		return self.ellsout.index(ell)
		
	@utils.classparams
	def set_grid(self,s,d=[0,scipy.inf]):
		for key in ['s']: setattr(self,key,scipy.array(self.params[key],dtype=self.TYPE_FLOAT))
		self.smask = (self.s >= d[0]) & (self.s <= d[-1])
		self.d = self.s[self.smask]

	def integrate_s(self,func):
		return integrate_trapz(func,self.s)
	
	def integrate_d(self,func):
		return integrate_trapz(func,self.d)

	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	@utils.loadstateclass
	def loadstate(self,state):
		self.setstate(state)
		return self
			
class ConvolvedIntegralConstraint(BaseIntegralConstraint):

	logger = logging.getLogger('ConvolvedIC')

	@utils.classparams
	def setup(self,s,k,ellsin,ellsout,path_window):
		self.set_window()
		self.set_grid()

	@utils.classparams
	def set_grid(self,s,k,ellsin,ellsout):

		super(ConvolvedIntegralConstraint,self).set_grid()
		
		weights = weights_trapz(self.d)
		def kernel(ellout):
			return scipy.asarray([1./(2*ellin[0]+1)*self.window[ellin[-1]]([self.s,self.d],(ellout,ellin[0]))*weights for ellin in self.ellsin])
		self.kernel = scipy.asarray([kernel(ellout) for ellout in self.ellsout])
		
		# self.real_shotnoise to be multiplied by the sample shot noise and subtracted from the correlation function
		if 'SN' in self.window:
			self.logger.info('Using provided shotnoise windows.')
			self.real_shotnoise = self.window['SN'](self.s,self.ellsout)
		else:
			self.logger.info('Shotnoise windows are inferred from full windows.')
			self.real_shotnoise = scipy.array([self.window[0]([self.s,0.],(ellout,0)) for ellout in self.ellsout])

	def ic(self,Xil):
		# to be subtracted from the correlation function
		Xil = Xil[:,self.smask]
		return scipy.sum(self.kernel*Xil[None,:,None,:],axis=(-3,-1))

	def getstate(self):
		state = {}
		for key in ['params','s','d','kernel','smask','los','ns','real_shotnoise']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','d','kernel','smask','los','ns','real_shotnoise']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state
			
class ConvolvedIntegralConstraintConvolution(BaseIntegralConstraint):

	logger = logging.getLogger('ConvolvedICConv')

	@utils.classparams
	def setup(self,s,k,ellsin,ellsconv,ellsout,path_window):
		self.set_window()
		self.set_grid()

	@utils.classparams
	def set_grid(self,s,k,ellsin,ellsconv,ellsout):

		super(ConvolvedIntegralConstraintConvolution,self).set_grid()
		weights = weights_trapz(self.d)
		def kernel(ellout):
			return scipy.asarray([1./(2*ellin[0]+1)*self.window[ellin[-1]]([self.s,self.d],(ellout,ellin[0]))*weights for ellin in self.ellsin])

		self.logger.info('Setting window function convolution correction.')
		self.kernel = MultipoleToMultipole(self.ellsconv,self.ellsout,lambda ell,n: kernel(ell)).conversion # shape is (self.ellsout,self.ellsconv,self.ellsin,self.s,self.d)

		if 'SN' in self.window:
			self.logger.info('Using provided shotnoise windows.')
			def kernel_shotnoise(ellout):
				return self.window['SN'](self.s,ellout)
		else:
			self.logger.info('Shotnoise windows are inferred from full windows.')
			def kernel_shotnoise(ellout):
				return self.window[0]([self.s,0.],(ellout,0))
		
		self.kernel_shotnoise = MultipoleToMultipole(self.ellsconv,self.ellsout,lambda ell,n: kernel_shotnoise(ell)).conversion # shape is (self.ellsout,self.ellsconv,self.s)
	
	def convolution(self,Xilc,Xilic):
		Xilic = Xilic[:,self.smask]
		return scipy.sum(scipy.sum(self.kernel*Xilic[None,:,None,:],axis=(-3,-1))*Xilc,axis=1) # first sum return (self.ellsout,self.ellsconv,self.s), second (self.ellsout,self.s)
	
	def real_shotnoise(self,Xilc):
		return scipy.sum(self.kernel_shotnoise*Xilc,axis=1)

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','d','kernel','smask','los','ns','kernel_shotnoise']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state
