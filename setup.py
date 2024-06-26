
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ncmcm',
    version='1.1.0',
    author='Akshey Kumar, Hofer Michael, Paul Eder, Emilija Mazuraite, Nenad Subat',
    author_email='akshey.kumar@univie.ac.at',
    description='A toolbox to visualize neuronal imaging data and apply the NC-MCM framework to it',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NC-MCM/NC-MCM',
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.10.12',
    install_requires=[
        'numpy==1.25.2',
        'matplotlib',
        'scikit-learn',
        'networkx',
        'pyvis',
        'statsmodels',
        'tensorflow==2.14.0',
        'typing_extensions==4.6.0',
        'tqdm',
        'scipy',
        'mat73',
        'seaborn',
        'pytest',
        #'markupsafe==2.0.1',
        'selenium',
        'jsonpickle',
    ],
)