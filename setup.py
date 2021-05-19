"""Tools and data for all-atom protein structure prediction via machine learning."""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open("README.md", "r") as handle:
    long_description = handle.read()

setup(
    # Self-descriptive entries which should always be present
    name='sidechainnet',
    author='Jonathan King',
    author_email='jok120@pitt.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='BSD-3-Clause',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    package_data={
        "astral_data": ["resources/astral_data.txt"],
        "full_protein_dssp": ["resources/full_protein_dssp_annotations.json"],
        "single_domain_dssp": ["resources/single_domain_dssp_annotations.json"]
        },

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    url='https://github.com/jonathanking/sidechainnet',  # Website
    install_requires=[
        'ProDy>=2.0', 'numpy', 'scipy', 'torch>=1.7', 'biopython', 'tqdm', 'py3Dmol',
        'requests', 'setuptools', 'pandas'
    ],  # Required packages, pulls from pip if needed; do not use for Conda deployment
    tests_require=['pytest'],
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.5",  # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ])
