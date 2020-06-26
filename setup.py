import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="checkerboard",
    version="0.2.3",
    author="Pierre Karashchuk",
    author_email="krchtchk@gmail.com",
    description="More robust checkerboard detection, similar algorithm to libcbdetect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lambdaloop/checkerboard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    install_requires=[
        'opencv-python'
    ],
    extras_require={
        'gpu':  ["gputools"]
    }
)
