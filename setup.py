import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py
import subprocess
from pathlib import Path
import os

def custom_command():
    subprocess.check_call(['make', '-C', 'musiclang_predict/c/'])


class CustomInstallBuildPy(build_py):
    """Custom build command that runs a Makefile."""

    def run(self):
        print('is called install')
        custom_command()
        # Call the superclass methods to handle Python extension building, if any
        build_py.run(self)


module = Extension('musiclang_predict.c',
                   sources=['musiclang_predict/c/run.c'],
                   include_dirs=[],
                   extra_compile_args=['-Ofast', '-fPIC', '-shared'])


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setuptools.setup(
    name="musiclang-predict",
    version="1.2.0",
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
    package_data={'musiclang_predict': ['c/*.h', 'c/*.c', 'c/*.so', 'c/*.dll', 'c/Makefile', 'corpus/*.mid'],
                  },
    include_package_data=True,
    python_requires=">=3.6",
)