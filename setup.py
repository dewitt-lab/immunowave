from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="immunowave",
    python_requires=">=3.9",
    version="0.1",
    description="Dynamical models for multi-scale immune responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author="William DeWitt",
    author_email="wsdewitt@gmail.com",
    install_requires=[
        "jaxlib",
        "jax",
        "diffrax",
        "matplotlib",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "flake8",
            "pytest",
            "black[jupyter]",
        ],
    },
)
