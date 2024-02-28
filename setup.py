import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import subprocess
import os

# Define the C extension
class CustomInstall(install):
    """Custom build command that runs a Makefile."""

    def run(self):
        # Check if we're on Windows
        # Run the Makefile
        print('is fucking called')
        subprocess.check_call(['make', '-C', 'musiclang_predict/c/'])
        # Call the superclass methods to handle Python extension building, if any
        install.run(self)

module = Extension('musiclang_predict.c',
                   sources=['musiclang_predict/c/run.c'],
                   include_dirs=[],
                   extra_compile_args=['-Ofast', '-fPIC', '-shared'])

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="musiclang-predict",
    version="0.0.1",
    author="Florian GARDIN",
    author_email="fgardin.pro@gmail.com",
    description=("Controllable symbolic music generation with generative AI"
                ),
    cmdclass={'install': CustomInstall},
    ext_modules=[module],
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        'Documentation': 'https://github.com/MusicLang/musiclang_predict',
        'Source': 'https://github.com/MusicLang/musiclang_predict',
        'Tracker': 'https://github.com/MusicLang/musiclang_predict/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
            "musiclang>=0.23",
            "torch",
            "transformers",
            "tokenizers",
            "torchtoolkit",
            "accelerate"
                      ],
    packages=setuptools.find_packages(include='*'),
    package_data={'musiclang_predict': ['c/*.h', 'c/*.so', 'c/*.dll']},
    include_package_data=True,
    python_requires=">=3.6",
)