import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ola-encoding",
    version="0.1.0",
    url="https://github.com/matsengrp/ola-encoding",
    author="Harry Richman",
    author_email="dhrichman@gmail.com",
    description="OLA encoding for phylogenetic trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["ola"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "ete3 >= 3.0.0",
        "pytest >= 7.3",
    ],
)
