import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob
with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config

setup(
    name='lensitbiases',
    version='0.0.1',
    packages=['lensitbiases'],
    url='https://github.com/NextGenCMB/lensitbiases',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='FFT-based lensing and other anisotropies N1-N0 bias calculator',
    data_files=[('lensitbiases/data/cls', ['lensitbiases/data/cls/FFP10_wdipole_lensedCls.dat',
                                         'lensitbiases/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                         'lensitbiases/data/cls/FFP10_wdipole_params.ini'])],
    install_requires=['numpy', 'pyfftw'],
    requires=['numpy', 'pyfftw'],
    long_description=long_description,
    configuration=configuration)

