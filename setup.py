from setuptools import setup, find_packages
from codecs import open
from os import path
from version import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='keeks',
    version=__version__,
    description='Python module for bankroll management',
    long_description=long_description,
    url='https://github.com/helton-tech/keeks',
    download_url='https://github.com/helton-tech/keeks/tarball/' + __version__,
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='elo scoring rating',
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    long_description_content_type="text/markdown",
    author='Will McGinnis',
    install_requires=[],
    author_email='will@helton.io'
)