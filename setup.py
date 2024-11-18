#!/usr/bin/env python

from setuptools import find_packages, setup

version = '0.0.4'
requires = [
    'boto3>=1.35.26'
]


setup(
    name='amazon-bedrock',
    version=version,
    description='A simplified Python client for Amazon Bedrock',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dennis Traub',
    url='https://github.com/boto/boto3',
    author_email='dennis.traub@gmail.com',
    package_dir={'': 'src'},  # Add this line
    packages=find_packages(where='src'),  # Modify this line
    install_requires=requires,
    license="MIT",
    python_requires=">= 3.8",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],
    project_urls={
        'Source': 'https://github.com/DennisTraub/amazon-bedrock-python',
    },
)