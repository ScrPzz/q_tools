from setuptools import find_packages, setup

setup(
    name='atogni_tools',
    packages=find_packages(include=['quant_tools', 'misc_tools', 'plot_tools', 'dict_tools']),
    version='0.1.5',
    description='Analysis tools',
    author='Alessandro Togni',
    license='MIT',
    python_requires='>=3'
)