from setuptools import setup, find_packages
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()
setup(
    name='GeneralAlphaZero',
    version='0.1.0',
    packages=find_packages(where='src'),  # Tell setuptools to find packages in 'src'
    package_dir={'': 'src'},  # Set 'src' as the package directory
    install_requires=read_requirements(),
    url='https://github.com/albinjal/GeneralAlphaZero',
    author='Albin Jaldevik',
    author_email='albin@jaldevik.com',
    description='General AlphaZero implementation for any game with a gym interface.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
