import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lfm_flow",
    version="0.0.2",
    author="Daniel Tait",
    author_email="daniel.tait@warwick.ac.uk",
    description="LFM in TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieljtait/lfm_flow",
    packages=setuptools.find_packages(exclude=['doc',]),
    install_requires=[
    'numpy',
    'scipy',
    'matplotlib>=2.2.0',
    'tensorflow>=1.13.0'],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    )
