"""
Setup script for EMBGuard package
Allows src modules to be imported as a package
"""
from setuptools import setup, find_packages

setup(
    name="embguard",
    version="0.1.0",
    description="EMBGuard safety guardrail system",
    license="MIT",
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        # Add your dependencies here if needed
        # "openai>=1.0.0",
        # "anthropic>=0.18.0",
        # etc.
    ],
)
