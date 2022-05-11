from setuptools import find_packages, setup

setup(
    name='q_tools',
    packages=find_packages(include=['q_tools']),
    version='0.1.0',
    description='Quantiles analysis tools',
    author='Alessandro Togni',
    license='MIT',
    install_requires=['pandas', 'numpy', 'matplotlib', ],
    python_requires='>=3'
)