from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filepath: str) -> List:
    with open(filepath, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements if req!=HYPHEN_E_DOT]
    return requirements

setup(
    name='income_prediction',
    version='0.0.1',
    description='Machine Learning Project',
    long_description=None,
    author='Dharanidharan',
    author_email='gdharanidharan07@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
