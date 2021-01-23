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
    description='Planck lensing python pipeline',
    install_requires=['numpy'], #removed mpi4py for travis tests
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)

