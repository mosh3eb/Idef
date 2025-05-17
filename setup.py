from setuptools import setup, find_packages

setup(
    name="idef",
    version="0.1.0",
    description="Interactive Data Exploration Framework for Multi-dimensional Scientific Datasets",
    author="IDEF Team",
    author_email="info@idef.org",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "xarray>=0.20.0",
        "dask>=2022.1.0",
        "plotly>=5.5.0",
        "holoviews>=1.14.0",
        "panel>=0.13.0",
        "param>=1.12.0",
        "bokeh>=2.4.0",
        "matplotlib>=3.5.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.1.0",
            "flake8>=4.0.0",
            "sphinx>=4.4.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
