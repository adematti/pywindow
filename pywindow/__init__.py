__version__ = '0.1'
__author__ = 'Arnaud de Mattia'
__all__ = ['catalogue','mu_function','window_function','utils']
__all__ += ['Catalogue','BaseCount','MuCount','MuFunction','PyReal2PCF','PyReal2PCFBinned','PyRealKMu','PyReal3PCF','PyReal4PCFBinned','Real2PCF','FFT2PCF','Real3PCFBinned','Real3PCFBinnedShotNoise','Real4PCFBinned',
'Real4PCFBinnedShotNoise','WindowFunction','WindowFunction1D','WindowFunction2D','setup_logging']
#__all__ += ['BaseAngularCount','Angular2PCF','Angular3PCFBinned','Analytic2PCF','Analytic3PCF','Analytic3PCFShotNoise','Analytic4PCF','Analytic4PCFShotNoise']

from .pyreal2pcf import PyReal2PCF,PyReal2PCFBinned,PyRealKMu
from .pyreal3pcf import PyReal3PCF
from .pyreal4pcf import PyReal4PCFBinned
from .correlation_catalogue import BaseCount,Real2PCF,FFT2PCF,Real3PCFBinned,Real3PCFBinnedShotNoise,Real4PCFBinned,Real4PCFBinnedShotNoise
#from .correlation_analytic import BaseAngularCount,Angular2PCF,Angular3PCFBinned,Analytic2PCF,Analytic3PCF,Analytic3PCFShotNoise,Analytic4PCF,Analytic4PCFShotNoise
from .window_function import WindowFunction,WindowFunction1D,WindowFunction2D,TemplateSystematics
from .mu_function import MuCount,MuFunction
from .catalogue import Catalogue
from .utils import setup_logging

