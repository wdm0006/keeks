from setuptools import setup, find_packages
from codecs import open
from os import path

VERSION = '0.0.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='keeks',
    version=VERSION,
    description='Python module for bankroll management',
    long_description=long_description,
    url='https://github.com/wdm0006/keeks',
    download_url='https://github.com/wdm0006/keeks/tarball/' + VERSION,
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='elo scoring rating',
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    author='Will McGinnis',
    install_requires=[],
    author_email='will@pedalwrencher.com'
)