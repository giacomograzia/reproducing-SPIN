"""Setup script for FACT AI reproducibility project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fact-ai-group-23",
    version="0.1.0",
    author="FACT AI Group 23",
    author_email="",
    description="Reproducibility project for UvA FACT AI course",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giacomograzia/fact-ai-group-23",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
)
