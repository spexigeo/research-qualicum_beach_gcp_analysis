"""Setup script for qualicum_beach_gcp_analysis package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="qualicum_beach_gcp_analysis",
    version="0.1.0",
    description="Tools for analyzing Qualicum Beach ground control points from KMZ files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mauricio Hess Flores",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "rasterio>=1.3.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)


from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="qualicum_beach_gcp_analysis",
    version="0.1.0",
    description="Tools for analyzing Qualicum Beach ground control points from KMZ files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mauricio Hess Flores",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "rasterio>=1.3.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)


from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="qualicum_beach_gcp_analysis",
    version="0.1.0",
    description="Tools for analyzing Qualicum Beach ground control points from KMZ files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mauricio Hess Flores",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "rasterio>=1.3.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

