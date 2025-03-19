import setuptools
from setuptools import setup, find_packages

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lensitbiases',
    version='0.0.1',
    packages=find_packages(),  # Automatically finds all packages
    url='https://github.com/NextGenCMB/lensitbiases',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='FFT-based lensing and other anisotropies N1-N0 bias calculator',
    install_requires=[
        'numpy',
        'pyfftw'
    ],
    include_package_data=True,  # Ensures data files are included
    package_data={
        'lensitbiases': [
            'data/cls/FFP10_wdipole_lensedCls.dat',
            'data/cls/FFP10_wdipole_lenspotentialCls.dat',
            'data/cls/FFP10_wdipole_params.ini'
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Ensures compatibility with your version
)
