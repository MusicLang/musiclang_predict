import setuptools

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="musiclang-predict",
    version="0.4.0",
    author="Florian GARDIN",
    author_email="fgardin.pro@gmail.com",
    description=("A python package for music generation using gen AI"
                ),
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
    include_package_data=True,
    python_requires=">=3.6",
)