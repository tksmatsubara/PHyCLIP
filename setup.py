from setuptools import find_packages, setup

setup(
    name="phyclip",
    version="1.0",
    python_requires=">=3.10",
    zip_safe=True,
    packages=find_packages(include=["phyclip"]),
)
