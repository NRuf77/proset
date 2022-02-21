"""Package installer configuration for proset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from setuptools import setup


with open("proset/__init__.py", mode="r", encoding="utf-8") as file:
    # note that we cannot just import the version from proset as dependencies may be missing
    content = file.readlines()
__version__ = None
for line in content:
    if line.startswith("__version__"):
        __version__ = line.split("=")[1].replace("\"", "").strip()
        break
if __version__ is None:
    raise RuntimeError("Unable to determine package version from proset/__init__.py.")

with open("README.md", mode="r", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="proset",
    version=__version__,
    description="Prototype set models for supervised learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Nikolaus Ruf",
    author_email="nikolaus.ruf@t-online.de",
    url="https://github.com/NRuf77/proset",
    packages=["proset", "proset.benchmarks", "proset.utility", "proset.utility.plots"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.3.2",
        "numpy>=1.19.2",
        "pandas>=1.1.3",
        "scipy>=1.5.2",
        "scikit-learn>=0.23.2",
        "statsmodels>=0.12.0"
    ],
    extras_require={"benchmarks": ["psutil>=5.7.2", "shap>=0.39.0", "xgboost>=1.3.3"]},
    exclude_package_data={
        "proset": ["__pycache__"],
        "proset.benchmarks": ["__pycache__"],
        "proset.utility": ["__pycache__"]
    }
)
