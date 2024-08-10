from setuptools import setup, find_packages

setup(
    name="ola-encoding",
    version="0.1.0",
    url="https://github.com/matsengrp/ola-encoding.git",
    author="Matsen Group",
    author_email="ematsen@gmail.com",
    description="OLA encoding for phylogenetic trees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "ete3 >= 3.0.0",
        "pytest >= 7.3",
    ],
    python_requires="==3.9.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
)
