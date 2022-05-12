from setuptools import find_packages, setup

setup(
    name='atogni_tools',
    packages=find_packages(include=['utilities']),
    version='0.1.9',
    description='Analysis tools',
    author='Alessandro Togni',
    license='MIT',
    python_requires='>=3'
)