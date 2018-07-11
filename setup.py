import os
import sys
import setuptools
import codecs

from setuptools.command.test import test as TestCommand


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

rust_requires = ['toml', 'setuptools_rust']
cuda_requires = []

tests_require = [
    'pytest', 
    'pytest-pythonpath', 
    'coverage', 
    'pytest-flake8'
    ] + rust_requires + cuda_requires

class PyTest(TestCommand):

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setuptools.setup(
    name='pxt',  # Required
    version='0.1.dev1',  # Required
    description='A small python package to easily compile code of other languages '
                'and link them to python code in a simple and flexible manner.',  # Required
    url='https://github.com/h3tch/pxt',  # Optional
    author='Michael Hecher',  # Optional
    author_email='michael.hecher@gmx.net',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='build compile link',  # Optional

    packages=setuptools.find_packages(exclude=['tests']),  # Required
    install_requires=['numpy'],  # Optional
    tests_require=tests_require, 
    extras_require={
        'rust': rust_requires,
        'cuda': cuda_requires,
    },
    zip_safe=False,
    cmdclass={'test': PyTest},
    package_data={  # Optional
        'pxt': ['pxt.h', 'npy.h', 'npy.cpp'],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/h3tch/pxt/issues',
        'Source': 'https://github.com/h3tch/pxt',
    },
)
