import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    config.add_extension('n1.n1f', ['n1/n1f.f90'],
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

