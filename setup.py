from setuptools import setup
from setuptools import find_packages

setup(name='rgcn',
      version='0.0.1',
      description='Graph Convolutional Networks for (directed) relational graphs',
      download_url='...',
      license='MIT',
      install_requires=['numpy',
                        'theano',
                        'keras',
                        'rdflib',
                        'scipy',
                        'pandas',
                        'wget'
                        ],
      extras_require={
          'model_saving': ['h5py'],
      },
      package_data={'rgcn': ['README.md', 'rgcn/data']},
      packages=find_packages())
