from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="factory-infection-model",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    python_requires=">=3.10",
)