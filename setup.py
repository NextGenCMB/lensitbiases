import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob
with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    fmods = glob.glob('n1/n1f*.f90')
    for fmod in fmods:
        name = 'n1.' + (fmod.split('/')[1]).replace('.f90', '')
        config.add_extension(name , [fmod],
                         libraries=['gomp'],  extra_compile_args=['-Xpreprocessor', '-fopenmp', '-w'])
    return config

setup(
    name='n1',
    version='0.0.1',
    packages=['n1'],
    url='https://github.com/carronj/n1',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='FFT-based lensing and other anisotropies N1 bias calculator',
    data_files=[('n1/data/cls', ['n1/data/cls/FFP10_wdipole_lensedCls.dat',
                                         'n1/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                         'n1/data/cls/FFP10_wdipole_params.ini'])],
    install_requires=['numpy'],
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)

