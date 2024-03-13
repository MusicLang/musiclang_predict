import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.build_clib import build_clib
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
import subprocess
import os

def custom_command():
    subprocess.check_call(['make', '-C', 'musiclang_predict/c/'])

# Define the C extension
class CustomInstall(install):
    """Custom build command that runs a Makefile."""

    def run(self):
        print('is called install')
        custom_command()
        # Call the superclass methods to handle Python extension building, if any
        install.run(self)

class CustomInstallClib(build_clib):
    """Custom build command that runs a Makefile."""

    def run(self):
        print('is called install')
        custom_command()
        # Call the superclass methods to handle Python extension building, if any
        build_clib.run(self)

class CustomInstallExt(install_lib):
    """Custom build command that runs a Makefile."""

    def run(self):
        print('is called install')
        custom_command()
        # Call the superclass methods to handle Python extension building, if any
        install_lib.run(self)

class CustomInstallBuildPy(build_py):
    """Custom build command that runs a Makefile."""

    def run(self):
        print('is called install')
        custom_command()
        # Call the superclass methods to handle Python extension building, if any
        build_py.run(self)


class CustomEggInfoCommand(egg_info):
    def run(self):
        print('is called egg')
        custom_command()
        egg_info.run(self)


module = Extension('musiclang_predict.c',
                   sources=['musiclang_predict/c/run.c'],
                   include_dirs=[],
                   extra_compile_args=['-Ofast', '-fPIC', '-shared'])

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setuptools.setup(
    name="musiclang-predict",
    version="1.1.6",
    author="Florian GARDIN",
    author_email="fgardin.pro@gmail.com",
    description=("Controllable symbolic music generation with generative AI"
                ),
    cmdclass={
                'build_py': CustomInstallBuildPy,
              },
    #ext_modules=[module],
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
            "musiclang>=0.25",
            "torch",
            "transformers",
            "tokenizers",
            "torchtoolkit",
            "accelerate"
                      ],
    packages=setuptools.find_packages(include='*'),
    package_data={'musiclang_predict': ['c/*.h', 'c/*.so', 'c/*.dll', 'c/Makefile', 'corpus/*.mid'],
                  },
    include_package_data=True,
    python_requires=">=3.6",
)