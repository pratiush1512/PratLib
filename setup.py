# setup.py
from setuptools import setup, find_packages

setup(
    name='pratlib',
    version='1.0.2',
    author='Pratiush',
    author_email='pratiushanand1@gmail.com',
    description='A PySpark library for machine learning, similar to scikit-learn.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pratiush1234/pratlib',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'pyspark>=3.0.0',
    ],
)
