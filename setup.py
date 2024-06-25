"""Setup"""

from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Image detection with PyTorch'
LONG_DESCRIPTION = 'Packge for detecting cars and number plates on images'

setup(
        name="wheresmycar",
        version=VERSION,
        author="Michal Dolbniak",
        author_email="md69626@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'torch',
            'torchvision',
        ],
        keywords=['python', 'pytorch']
)
