from setuptools import find_packages
from setuptools import setup

setup(
    name='procedure_cloning',
    version='0.0',
    description=(
        'Procedure cloning for chain of thought imitation learning.'
    ),
    packages=find_packages(),
    package_data={},
    install_requires=[
        'pytorch',
        'numpy',
        'dice_rl',
    ])
