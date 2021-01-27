import os
import sys
from setuptools import setup

setup_keywords = dict()
setup_keywords['name'] = 'pywindow'
setup_keywords['description'] = 'Window function utility'
setup_keywords['author'] = 'Arnaud de Mattia'
setup_keywords['author_email'] = ''
setup_keywords['license'] = 'GPL3'
setup_keywords['url'] = 'https://github.com/adematti/pywindow'
sys.path.insert(0,os.path.abspath('pywindow/'))
import version
setup_keywords['version'] = version.__version__
setup_keywords['install_requires'] = ['numpy','scipy']
setup_keywords['packages'] = ['pywindow']
setup_keywords['package_dir'] = {'pywindow':'pywindow'}

setup_keywords['cmdclass'] = {}

setup(**setup_keywords)
