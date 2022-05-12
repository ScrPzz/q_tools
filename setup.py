from setuptools import find_packages, setup

setup(
    name='atogni_tools',
    packages=find_packages(include=['dict_tools', 'misc_tools', 'plot_tools', 'quant_tools']),
    version='0.2.0',
    description='Analysis tools',
    author='Alessandro Togni',
    license='MIT',
    python_requires='>=3'
)