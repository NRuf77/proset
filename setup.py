"""Package installer configuration for proset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from setuptools import setup


with open("proset/__init__.py", mode="r", encoding="utf-8") as file:
    # retrieve module version without importing the package
    version = file.readlines()
version = [line for line in version if line.startswith("__version__")][0]
version = version.split("=")[1].replace("\"", "").strip()

with open("README.md", mode="r", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="proset",
    version=version,
    description="Prototype set models for supervised learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Nikolaus Ruf",
    author_email="nikolaus.ruf@t-online.de",
    url="https://github.com/NRuf77/proset",
    packages=["proset", "proset.benchmarks", "proset.objectives", "proset.utility", "proset.utility.plots"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.8.3",
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.1.post1",
        "statsmodels>=0.14.1",
        "xlsxwriter>=3.1.9"
    ],
    extras_require={
        "benchmarks": ["mnist>=0.2.2", "psutil>=5.9.8", "shap>=0.44.0", "xgboost>=2.0.3"],
        "development": ["coverage>=7.4.3", "ipython>=8.22.1", "pylint>=3.0.4", "twine>=5.0.0"],
        "tensorflow": ["tensorflow>=2.15.0"]
    },
    exclude_package_data={
        "proset": ["__pycache__"],
        "proset.benchmarks": ["__pycache__"],
        "proset.objectives": ["__pycache__"],
        "proset.utility": ["__pycache__"]
    }
)
