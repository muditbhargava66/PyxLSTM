from setuptools import setup, find_packages

setup(
    name="PyxLSTM",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0.0",
    ],
    author="Mudit Bhargava",
    author_email="muditbhargava666@gmail.com",
    description="PyxLSTM: An efficient and extensible implementation of the xLSTM architecture",
    keywords="xLSTM, LSTM, language modeling, sequence modeling",
    url="https://github.com/muditbhargava66/PyxLSTM.git",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)