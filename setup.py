import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-distbelief",
    version="0.1.0",
    author="Jesse Cai",
    author_email="jcjessecai@gmail.com",
    description="Distributed training for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ucla-labx/distbelief",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
